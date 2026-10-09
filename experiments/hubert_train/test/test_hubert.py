import argparse
import csv
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torchaudio
from sklearn.cluster import MiniBatchKMeans

from experiments.hubert_train.extract_features import NpyStreamWriter, run_extract
from experiments.hubert_train.finetune_hubert import apply_preset
from src.data.duration_cache import build_duration_caches, dataset_audio_paths, open_subset
from src.data.hubert_data_module import (
    get_hubert_finetune_data_module,
    get_hubert_pretrain_data_module,
)
from src.data.hubert_transforms import (
    HUBERT_BLANK_ID,
    HUBERT_LTR_TO_ID,
    HUBERT_LTR_VOCAB,
    DummyHubertPretrainTransform,
    HubertCharFinetuneTransform,
    decode_hubert_ltr,
    encode_hubert_ltr,
    load_km_file,
    librispeech_utt_id,
)
from src.models.hubert.config import get_hubert_config, hubert_base, hubert_tiny
from src.models.hubert.hubert_model import HubertModel
from src.models.hubert.kmeans import (
    ProgressMiniBatchKMeans,
    default_fit_splits,
    extract_mfcc_39,
    fit_kmeans,
    predict_labels,
    sample_frames,
    sample_row_ranges,
    split_labels,
    split_row_ranges,
)
from src.models.hubert.masking import apply_mask, compute_channel_mask, compute_span_mask
from src.models.hubert.prediction_head import HubertPredictionHead
from src.models.hubert_lightning_module import HubertCTCModule, HubertPretrainModule
from src.opt.schedulers import LinearWarmupDecayScheduler, TriStageLRScheduler

def _ctc_batch():
    return argparse.Namespace(
        inputs=torch.randn(2, 6400),
        input_lengths=torch.tensor([6400, 4800], dtype=torch.long),
        targets=torch.tensor([[4, 5, 6], [7, 8, 27]], dtype=torch.long),
        target_lengths=torch.tensor([3, 3], dtype=torch.long),
    )

def _fake_librispeech_batches(n_batches, batch_size, seconds=0.5, seed=0):
    gen = torch.Generator().manual_seed(seed)
    batches = []
    for b in range(n_batches):
        batches.append([
            (torch.randn(1, int(16000 * seconds), generator=gen), 16000, "A B", 1, 2, b * batch_size + i)
            for i in range(batch_size)
        ])
    return batches

def _tiny_batch(batch_size=2, samples=3200, num_classes=16, label_rate=50.0):
    source = torch.randn(batch_size, samples)
    lengths = torch.tensor([samples, samples // 2], dtype=torch.long)
    hop = int(16000 / label_rate)
    n_lab = max(samples // hop, 8)
    targets = torch.randint(0, num_classes, (batch_size, n_lab))
    return source, lengths, [targets]

class TestCnnEncoder(unittest.TestCase):
    def test_output_length_matches_stacked_convs(self):

        cfg = hubert_tiny()
        model = HubertModel(cfg)
        lengths = torch.tensor([16000, 8000, 3200], dtype=torch.long)
        out = model.feature_extractor.output_lengths(lengths)
        expected = lengths.clone()
        for _, kernel, stride in cfg.conv_layers:
            expected = torch.div(expected - kernel, stride, rounding_mode="floor") + 1
            expected = expected.clamp_min(0)
        self.assertTrue(torch.equal(out, expected))

    def test_forward_time_axis_equals_output_lengths(self):

        model = HubertModel(hubert_tiny())
        source = torch.randn(2, 16000)
        lengths = torch.tensor([16000, 12000], dtype=torch.long)
        feats = model.feature_extractor(source)
        feat_lengths = model.feature_extractor.output_lengths(lengths)
        self.assertEqual(feats.shape[2], int(feat_lengths.max().item()))
        self.assertEqual(feats.shape[1], 32)

class TestMasking(unittest.TestCase):
    def test_mask_stays_inside_valid_frames(self):

        lengths = torch.tensor([40, 11, 3], dtype=torch.long)
        mask = compute_span_mask(lengths, mask_prob=0.08, mask_length=10, min_masks=2)
        for i, length in enumerate(lengths.tolist()):
            self.assertFalse(mask[i, length:].any())
            if length > 1:
                self.assertTrue(mask[i, :length].any())

    def test_mask_uses_span_length(self):

        lengths = torch.tensor([80], dtype=torch.long)
        mask = compute_span_mask(lengths, mask_prob=0.08, mask_length=10, min_masks=2)
        idx = torch.where(mask[0])[0].tolist()
        self.assertGreaterEqual(len(idx), 10)

    def test_apply_mask_writes_embedding(self):

        x = torch.zeros(1, 5, 4)
        mask = torch.tensor([[False, True, True, False, False]])
        emb = torch.ones(4)
        y = apply_mask(x, mask, emb)
        self.assertTrue(torch.allclose(y[0, 1], emb))
        self.assertTrue(torch.allclose(y[0, 0], torch.zeros(4)))

class TestPredictionHead(unittest.TestCase):
    def test_logits_shape_and_temperature(self):

        head = HubertPredictionHead(embed_dim=8, final_dim=4, num_classes=[10], logit_temp=0.1)
        hidden = torch.randn(6, 8)
        logits = head.logits(hidden, 0)
        self.assertEqual(tuple(logits.shape), (6, 10))

    def test_ensemble_returns_one_logit_tensor_per_codebook(self):

        head = HubertPredictionHead(embed_dim=8, final_dim=4, num_classes=[5, 7])
        hidden = torch.randn(3, 8)
        outs = head(hidden)
        self.assertEqual(len(outs), 2)
        self.assertEqual(tuple(outs[0].shape), (3, 5))
        self.assertEqual(tuple(outs[1].shape), (3, 7))

class TestHubertMaskedLoss(unittest.TestCase):
    def test_alpha_one_matches_masked_cross_entropy(self):

        cfg = hubert_tiny(num_classes=[16], label_rate=50.0, mask_alpha=1.0)
        model = HubertModel(cfg)
        source, lengths, targets = _tiny_batch()
        torch.manual_seed(0)
        out = model(source, lengths, targets, mask=True)
        loss = model.masked_prediction_loss(out)
        manual = torch.nn.functional.cross_entropy(
            out["logit_m_list"][0].float(), out["target_m_list"][0].long()
        )
        manual = manual + cfg.feature_penalty_weight * out["features_pen"]
        self.assertTrue(torch.allclose(loss, manual, atol=1e-5))

    def test_alpha_zero_matches_unmasked_cross_entropy(self):

        cfg = hubert_tiny(num_classes=[16], label_rate=50.0, mask_alpha=0.0)
        model = HubertModel(cfg)
        source, lengths, targets = _tiny_batch(samples=16000)
        torch.manual_seed(1)
        out = model(source, lengths, targets, mask=True)
        self.assertGreater(out["logit_u_list"][0].numel(), 0)
        loss = model.masked_prediction_loss(out)
        manual = torch.nn.functional.cross_entropy(
            out["logit_u_list"][0].float(), out["target_u_list"][0].long()
        )
        manual = manual + cfg.feature_penalty_weight * out["features_pen"]
        self.assertTrue(torch.allclose(loss, manual, atol=1e-5))

    def test_padding_excluded_from_masked_targets(self):

        cfg = hubert_tiny(num_classes=[16], label_rate=50.0)
        model = HubertModel(cfg)
        source, lengths, targets = _tiny_batch()
        out = model(source, lengths, targets, mask=True)
        self.assertFalse((out["mask_indices"] & out["padding_mask"]).any())
        n_valid = int((~out["padding_mask"]).sum().item())
        n_m = out["target_m_list"][0].numel()
        n_u = out["target_u_list"][0].numel()
        self.assertEqual(n_m + n_u, n_valid)

    def test_ensemble_loss_averages_codebooks(self):

        cfg = hubert_tiny(num_classes=[8, 12], label_rate=50.0, mask_alpha=1.0)
        model = HubertModel(cfg)
        source = torch.randn(2, 3200)
        lengths = torch.tensor([3200, 3200], dtype=torch.long)
        n_lab = 40
        targets = [
            torch.randint(0, 8, (2, n_lab)),
            torch.randint(0, 12, (2, n_lab)),
        ]
        out = model(source, lengths, targets, mask=True)
        loss = model.masked_prediction_loss(out)
        parts = [
            torch.nn.functional.cross_entropy(lm.float(), tm.long())
            for lm, tm in zip(out["logit_m_list"], out["target_m_list"])
        ]
        manual = torch.stack(parts).mean() + cfg.feature_penalty_weight * out["features_pen"]
        self.assertTrue(torch.allclose(loss, manual, atol=1e-5))

class TestAlignAndLayers(unittest.TestCase):
    def test_label_alignment_subsamples_100hz_to_cnn_rate(self):

        cfg = hubert_tiny(num_classes=[16], label_rate=100.0)
        model = HubertModel(cfg)
        self.assertAlmostEqual(model.feat2tar_ratio, 2.0)
        features = torch.randn(1, 32, 10)
        labels = torch.arange(20).view(1, 20)
        _, aligned = model.align_targets(features, [labels])
        self.assertEqual(aligned[0].shape[-1], 10)
        self.assertEqual(aligned[0][0, 1].item(), 2)

    def test_extract_features_returns_requested_layer(self):

        model = HubertModel(hubert_tiny())
        source = torch.randn(1, 3200)
        lengths = torch.tensor([3200], dtype=torch.long)
        x0, _, _ = model.extract_features(source, lengths, tgt_layer=0)
        x1, _, _ = model.extract_features(source, lengths, tgt_layer=1)
        self.assertEqual(x0.shape, x1.shape)
        self.assertFalse(torch.allclose(x0, x1))

class TestKmeansAndMfcc(unittest.TestCase):
    def test_mfcc_has_39_dimensions(self):

        wav = torch.sin(torch.linspace(0, 200, 16000))
        feats = extract_mfcc_39(wav, 16000)
        self.assertEqual(feats.shape[1], 39)
        self.assertGreater(feats.shape[0], 10)

    def test_kmeans_fit_and_predict_roundtrip(self):

        rng = np.random.RandomState(0)
        a = rng.randn(80, 4) + np.array([5.0, 0, 0, 0])
        b = rng.randn(80, 4) + np.array([-5.0, 0, 0, 0])
        feats = np.concatenate([a, b], axis=0).astype(np.float32)
        km = fit_kmeans(feats, n_clusters=2, percent=1.0, n_init=1, max_iter=20, seed=0)
        labels = predict_labels(km, feats)
        self.assertEqual(labels.shape[0], 160)
        self.assertEqual(len(set(labels.tolist())), 2)

    def test_split_labels_restores_utterance_boundaries(self):

        split = split_labels([1, 2, 3, 4, 5], [2, 3])
        self.assertEqual(split, [[1, 2], [3, 4, 5]])

    def test_load_km_file_parses_utt_and_codes(self):

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.km"
            path.write_text("19-198-0001 0 1 1 2\n19-198-0002 3 3\n", encoding="utf-8")
            loaded = load_km_file(str(path))
        self.assertEqual(loaded["19-198-0001"], [0, 1, 1, 2])
        self.assertEqual(loaded["19-198-0002"], [3, 3])

class TestConfigAndTransforms(unittest.TestCase):
    def test_base_config_matches_paper_table(self):

        cfg = hubert_base(num_classes=[100])
        self.assertEqual(cfg.encoder_layers, 12)
        self.assertEqual(cfg.encoder_embed_dim, 768)
        self.assertEqual(cfg.encoder_ffn_embed_dim, 3072)
        self.assertEqual(cfg.encoder_attention_heads, 12)
        self.assertEqual(cfg.final_dim, 256)
        self.assertEqual(cfg.conv_layers, [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ])
        self.assertEqual(cfg.mask_prob, 0.08)
        self.assertEqual(cfg.mask_length, 10)
        self.assertEqual(cfg.feature_grad_mult, 0.1)

    def test_get_hubert_config_rejects_unknown_name(self):

        with self.assertRaises(ValueError):
            get_hubert_config("huge")

    def test_dummy_pretrain_transform_builds_padded_batch(self):

        transform = DummyHubertPretrainTransform(num_classes=[4], label_rate=50.0)
        wav1 = torch.randn(1, 16000)
        wav2 = torch.randn(1, 8000)
        samples = [
            (wav1, 16000, "a", 1, 2, 3),
            (wav2, 16000, "b", 1, 2, 4),
        ]
        batch = transform(samples)
        self.assertEqual(batch.inputs.shape[0], 2)
        self.assertEqual(int(batch.input_lengths[0]), 16000)
        self.assertEqual(int(batch.input_lengths[1]), 8000)
        self.assertEqual(batch.targets[0].shape[0], 2)

    def test_utt_id_format(self):

        sample = (torch.zeros(1, 10), 16000, "hi", 19, 198, 1)
        self.assertEqual(librispeech_utt_id(sample), "19-198-0001")

class TestTrainingStep(unittest.TestCase):
    def test_tiny_forward_backward(self):

        cfg = hubert_tiny(num_classes=[16], label_rate=50.0)
        model = HubertModel(cfg)
        source, lengths, targets = _tiny_batch()
        loss = model.masked_prediction_loss(model(source, lengths, targets, mask=True))
        loss.backward()
        grads = [p.grad.abs().sum() for p in model.parameters() if p.grad is not None]
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(len(grads), 0)
        self.assertGreater(sum(float(g) for g in grads), 0.0)

    def test_pretrain_module_step(self):

        args = argparse.Namespace(
            model_size="tiny",
            num_classes="16",
            label_rate=50.0,
            mask_alpha=1.0,
            lr=1e-3,
            weight_decay=0.0,
            warmup_ratio=0.08,
            max_steps=20,
        )
        module = HubertPretrainModule(args)
        module.log = lambda *a, **k: None
        source, lengths, targets = _tiny_batch()
        batch = argparse.Namespace(
            inputs=source,
            input_lengths=lengths,
            targets=targets,
        )
        loss = module._step(batch, "train")
        self.assertTrue(torch.isfinite(loss))

    def test_freeze_feature_extractor(self):

        model = HubertModel(hubert_tiny())
        model.freeze_feature_extractor()
        for p in model.feature_extractor.parameters():
            self.assertFalse(p.requires_grad)
        self.assertEqual(model.feature_grad_mult, 0.0)

    def test_remove_pretraining_modules_drops_head(self):

        model = HubertModel(hubert_tiny())
        model.remove_pretraining_modules()
        self.assertIsNone(model.pred_head)

    def test_ctc_module_greedy_decode_and_cnn_frozen(self):

        class FakeSP:
            def get_piece_size(self):
                return 128

            def decode(self, ids):
                return " ".join(str(i) for i in ids)

        args = argparse.Namespace(
            model_size="tiny",
            num_classes="16",
            label_rate=50.0,
            mask_alpha=1.0,
            lr=1e-3,
            freeze_steps=0,
            warmup_steps=10,
            pretrained_path=None,
        )
        module = HubertCTCModule(args, FakeSP())
        module.log = lambda *a, **k: None
        for p in module.encoder.feature_extractor.parameters():
            self.assertFalse(p.requires_grad)
        wav = torch.randn(1, 3200)
        lengths = torch.tensor([3200], dtype=torch.long)
        targets = torch.tensor([[4, 5, 6]], dtype=torch.int32)
        target_lengths = torch.tensor([3], dtype=torch.int32)
        batch = argparse.Namespace(
            inputs=wav,
            input_lengths=lengths,
            targets=targets,
            target_lengths=target_lengths,
        )
        loss = module._step(batch, "train")
        text = module(batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsInstance(text, str)

    def test_freeze_steps_keeps_transformer_frozen(self):

        class FakeSP:
            def get_piece_size(self):
                return 128

            def decode(self, ids):
                return ""

        args = argparse.Namespace(
            model_size="tiny",
            num_classes="16",
            label_rate=50.0,
            mask_alpha=1.0,
            lr=1e-3,
            freeze_steps=5,
            warmup_steps=10,
            pretrained_path=None,
        )
        module = HubertCTCModule(args, FakeSP())
        module.log = lambda *a, **k: None
        transformer = [
            p for n, p in module.encoder.named_parameters() if not n.startswith("feature_extractor")
        ]
        self.assertTrue(transformer)
        self.assertTrue(all(p.requires_grad for p in transformer))

        batch = _ctc_batch()
        with mock.patch.object(type(module), "global_step", new_callable=mock.PropertyMock, return_value=0):
            module._step(batch, "train").backward()
        self.assertTrue(all(p.grad is None for p in transformer))
        self.assertIsNotNone(module.ctc_out.weight.grad)

        module.zero_grad(set_to_none=True)
        with mock.patch.object(type(module), "global_step", new_callable=mock.PropertyMock, return_value=5):
            module._step(batch, "train").backward()
        self.assertTrue(any(p.grad is not None for p in transformer))

    def test_gradual_unfreeze_releases_layers_top_down(self):

        args = argparse.Namespace(
            model_size="tiny", num_classes="16", label_rate=50.0, mask_alpha=1.0, lr=1e-3,
            freeze_steps=10, unfreeze_steps=30, warmup_steps=10, pretrained_path=None,
        )
        module = HubertCTCModule(args)
        total = len(module._unfreeze_groups)
        self.assertEqual(total, module.encoder.cfg.encoder_layers + 1)

        def frozen_at(step):
            with mock.patch.object(type(module), "global_step", new_callable=mock.PropertyMock, return_value=step):
                return module._frozen_groups()

        self.assertEqual(frozen_at(0), total)
        self.assertEqual(frozen_at(10), total - 1)
        self.assertEqual(frozen_at(39), 0)
        self.assertEqual(frozen_at(40), 0)

        for p in module.parameters():
            p.grad = torch.ones_like(p)
        with mock.patch.object(type(module), "global_step", new_callable=mock.PropertyMock, return_value=10):
            module.on_before_optimizer_step(module.optimizer)
        top = module._unfreeze_groups[-1]
        bottom = module._unfreeze_groups[0]
        self.assertTrue(all(p.grad is not None for p in top))
        self.assertTrue(all(p.grad is None for p in bottom))
        self.assertIsNotNone(module.ctc_out.weight.grad)

class TestPretrainedInit(unittest.TestCase):
    def _args(self, **over):
        base = dict(
            model_size="tiny", num_classes="16", label_rate=50.0, mask_alpha=1.0, lr=1e-3,
            freeze_steps=0, warmup_steps=10, pretrained_path=None,
        )
        base.update(over)
        return argparse.Namespace(**base)

    def _save_pretrain_ckpt(self, path, model_size="tiny", global_step=100, max_steps=1000):
        torch.manual_seed(123)
        module = HubertPretrainModule(argparse.Namespace(
            model_size=model_size, num_classes="16", label_rate=50.0, mask_alpha=1.0,
            lr=1e-3, weight_decay=0.0, warmup_ratio=0.08, max_steps=max_steps,
        ))
        torch.save(
            {
                "state_dict": module.state_dict(),
                "hyper_parameters": {"model_size": model_size, "max_steps": max_steps},
                "global_step": global_step,
                "epoch": 3,
            },
            path,
        )
        return module

    def test_pretrained_weights_are_loaded_and_reported(self):

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pt.ckpt"
            pretrain = self._save_pretrain_ckpt(path)
            module = HubertCTCModule(self._args(pretrained_path=path))
        for name, value in module.encoder.state_dict().items():
            self.assertTrue(torch.equal(value, pretrain.model.state_dict()[name]), name)
        self.assertEqual(module.hparams.pretrained_global_step, 100)
        self.assertEqual(
            module.hparams.pretrained_loaded_tensors,
            len([k for k in pretrain.model.state_dict() if not k.startswith("pred_head.")]),
        )

    def test_size_mismatch_raises_instead_of_silent_partial_load(self):

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pt.ckpt"
            self._save_pretrain_ckpt(path, model_size="tiny")
            with self.assertRaisesRegex(ValueError, "does not match the encoder"):
                HubertCTCModule(self._args(model_size="small", pretrained_path=path))

    def test_random_init_never_loads_a_checkpoint(self):

        with mock.patch(
            "src.models.hubert_lightning_module.load_encoder_state",
            side_effect=AssertionError("pretrained weights must not be loaded"),
        ):
            module = HubertCTCModule(self._args(random_init=True))
        self.assertTrue(module.random_init)
        self.assertFalse(hasattr(module.hparams, "pretrained_global_step"))
        self.assertTrue(all(p.requires_grad for p in module.encoder.feature_extractor.parameters()))
        self.assertGreater(module.encoder.feature_grad_mult, 0.0)
        cnn_ids = {id(p) for p in module.encoder.feature_extractor.parameters()}
        opt_ids = {id(p) for g in module.optimizer.param_groups for p in g["params"]}
        self.assertTrue(cnn_ids <= opt_ids)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pt.ckpt"
            pretrain = self._save_pretrain_ckpt(path)
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                HubertCTCModule(self._args(random_init=True, pretrained_path=path))
            torch.manual_seed(0)
            scratch = HubertCTCModule(self._args(random_init=True))
        pt_state = pretrain.model.state_dict()
        differs = [
            not torch.equal(v, pt_state[k])
            for k, v in scratch.encoder.state_dict().items()
            if k.endswith("weight") and v.dim() > 1
        ]
        self.assertTrue(all(differs))

    def test_eval_reload_skips_pretrained_path(self):

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pt.ckpt"
            self._save_pretrain_ckpt(path)
            module = HubertCTCModule(self._args(pretrained_path=path))
            path.unlink()
            reloaded = HubertCTCModule(module.hparams, load_pretrained=False)
        self.assertIsNotNone(reloaded)

class TestFinetuneLabels(unittest.TestCase):
    def test_ltr_vocab_matches_fairseq_letters(self):

        self.assertEqual(len(HUBERT_LTR_VOCAB), 28)
        self.assertEqual(HUBERT_BLANK_ID, 28)
        self.assertEqual(HUBERT_LTR_VOCAB[-1], "|")

    def test_ltr_roundtrip_restores_transcript(self):

        ids = encode_hubert_ltr("the cat's hat")
        self.assertEqual(decode_hubert_ltr(ids), "THE CAT'S HAT")

    def test_ltr_appends_word_boundary(self):

        ids = encode_hubert_ltr("ok")
        self.assertEqual(ids[-1], HUBERT_LTR_TO_ID["|"])

    def test_char_finetune_transform_pads_with_blank(self):

        transform = HubertCharFinetuneTransform()
        samples = [
            (torch.randn(1, 3200), 16000, "hi", 1, 2, 3),
            (torch.randn(1, 1600), 16000, "a b", 1, 2, 4),
        ]
        batch = transform(samples)
        self.assertEqual(batch.targets.shape[0], 2)
        self.assertEqual(int(batch.target_lengths[0]), len(encode_hubert_ltr("hi")))
        pad_len = int(batch.targets.shape[1] - batch.target_lengths[1])
        if pad_len:
            self.assertTrue((batch.targets[1, -pad_len:] == HUBERT_BLANK_ID).all())

class TestFinetuneMasking(unittest.TestCase):
    def test_channel_mask_uses_fairseq_span_count(self):

        mask = compute_channel_mask(4, 32, mask_prob=0.5, mask_length=8, device=torch.device("cpu"))
        self.assertEqual(tuple(mask.shape), (4, 32))
        for row in mask:
            n_masked = int(row.sum())
            self.assertGreaterEqual(n_masked, 8)
            self.assertLessEqual(n_masked, 3 * 8)

    def test_channel_mask_off_when_prob_zero(self):

        mask = compute_channel_mask(2, 32, mask_prob=0.0, mask_length=8, device=torch.device("cpu"))
        self.assertFalse(mask.any())

    def test_extract_features_masks_only_when_asked(self):

        torch.manual_seed(0)
        model = HubertModel(hubert_tiny()).eval()
        source = torch.randn(2, 6400)
        lengths = torch.tensor([6400, 4800], dtype=torch.long)
        with torch.no_grad():
            plain, plain_lengths, _ = model.extract_features(source, lengths)
            torch.manual_seed(0)
            masked, masked_lengths, _ = model.extract_features(
                source, lengths, mask=True, mask_prob=0.2, mask_channel_prob=0.5
            )
        self.assertTrue(torch.equal(plain_lengths, masked_lengths))
        self.assertEqual(plain.shape, masked.shape)
        self.assertFalse(torch.allclose(plain, masked))

    def test_set_finetune_dropout_overrides_every_layer(self):

        model = HubertModel(hubert_base(num_classes=[100]))
        model.set_finetune_dropout(
            dropout=0.0, attention_dropout=0.0, activation_dropout=0.1, layerdrop=0.1
        )
        self.assertEqual(model.encoder.layerdrop, 0.1)
        for layer in model.encoder.layers:
            self.assertEqual(layer.dropout, 0.0)
            self.assertEqual(layer.activation_dropout, 0.1)
            self.assertEqual(layer.self_attn.dropout, 0.0)

class TestKmeansProgress(unittest.TestCase):
    def test_progress_subclass_matches_plain_minibatch_kmeans(self):

        rng = np.random.RandomState(0)
        feats = np.concatenate(
            [rng.randn(200, 4) + 5.0, rng.randn(200, 4) - 5.0]
        ).astype(np.float32)
        kwargs = dict(
            n_clusters=2,
            init="k-means++",
            max_iter=10,
            batch_size=50,
            compute_labels=False,
            tol=0.0,
            max_no_improvement=100,
            n_init=2,
            reassignment_ratio=0.0,
            random_state=0,
        )
        ours = ProgressMiniBatchKMeans(**kwargs).fit(feats)
        theirs = MiniBatchKMeans(**kwargs).fit(feats)
        self.assertTrue(np.allclose(ours.cluster_centers_, theirs.cluster_centers_))

    def test_fitted_model_survives_pickle_with_bar_attached(self):

        rng = np.random.RandomState(0)
        feats = rng.randn(200, 4).astype(np.float32)
        model = fit_kmeans(feats, n_clusters=3, n_init=1, max_iter=5, seed=0)
        model._bar = object()
        restored = pickle.loads(pickle.dumps(model))
        self.assertIsNone(restored._bar)
        self.assertTrue(np.allclose(restored.cluster_centers_, model.cluster_centers_))

    def test_sample_frames_caps_at_max_frames(self):

        rng = np.random.RandomState(0)
        feats = rng.randn(5000, 6).astype(np.float32)
        out = sample_frames(feats, percent=1.0, rng=rng, max_frames=1000, chunk=256)
        self.assertEqual(out.shape, (1000, 6))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(out.flags["C_CONTIGUOUS"])

    def test_sample_frames_returns_everything_without_cap(self):

        rng = np.random.RandomState(0)
        feats = rng.randn(500, 6).astype(np.float32)
        out = sample_frames(feats, percent=1.0, rng=rng, max_frames=0)
        self.assertEqual(out.shape, feats.shape)

def _write_fake_chapter(chapter_dir, speaker, chapter, n_utts, secs=1.0):
    chapter_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for u in range(n_utts):
        fileid = f"{speaker}-{chapter}-{u:04d}"
        wav = torch.randn(1, int(secs * 16000)) * 0.05
        torchaudio.save(str(chapter_dir / f"{fileid}.flac"), wav, 16000)
        lines.append(f"{fileid} HELLO WORLD {u}")
    (chapter_dir / f"{speaker}-{chapter}.trans.txt").write_text("\n".join(lines) + "\n")

def _fake_corpora(root):

    for split in ("train-clean-100", "train-clean-360", "dev-clean", "dev-other"):
        _write_fake_chapter(root / "LibriSpeech" / split / "19" / "198", 19, 198, 3)
    ll = root / "librispeech_finetuning"
    for k in range(6):
        _write_fake_chapter(ll / "1h" / str(k) / "clean" / "103" / "1240", 103, 1240, 1)
    _write_fake_chapter(ll / "9h" / "clean" / "103" / "1241", 103, 1241, 2)
    _write_fake_chapter(ll / "9h" / "other" / "104" / "1300", 104, 1300, 2)

class TestSubsets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        _fake_corpora(cls.root)
        build_duration_caches(
            str(cls.root),
            ["train-clean-100", "train-clean-360", "dev-clean", "dev-other", "ll-10h", "ll-1h"],
            str(cls.root / ".duration_cache"),
            scan_workers=1,
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_librilight_10h_includes_all_1h_folds_and_9h(self):

        self.assertEqual(len(open_subset(self.root, "ll-10h")), 6 + 2 + 2)
        self.assertEqual(len(open_subset(self.root, "ll-1h")), 6)

    def test_librilight_duration_paths_point_at_real_files(self):

        ds = open_subset(self.root, "ll-10h")
        paths = dataset_audio_paths(ds)
        self.assertEqual(len(paths), len(ds))
        self.assertTrue(all(Path(p).is_file() for p in paths))
        wav, sr, text, *_ = ds[0]
        self.assertEqual(sr, 16000)
        self.assertTrue(text.startswith("HELLO WORLD"))

    def test_pretrain_uses_only_requested_subsets(self):

        dm = get_hubert_pretrain_data_module(
            str(self.root),
            dummy_labels=True,
            num_classes=[16],
            label_rate=100.0,
            num_workers=0,
            max_batch_duration=20.0,
            train_subsets=["train-clean-100"],
        )
        loader = dm.train_dataloader()
        n_utts = sum(len(b) for b in loader.dataset.dataset.datasets[0].batches)
        self.assertEqual(len(loader.dataset.dataset.datasets), 1)
        self.assertEqual(n_utts, 3)
        batch = next(iter(loader))
        self.assertEqual(batch.inputs.shape[0], len(batch.input_lengths))

    def test_finetune_on_librilight_10h_builds_char_batches(self):

        dm = get_hubert_finetune_data_module(
            str(self.root),
            label_type="char",
            num_workers=0,
            max_batch_duration=50.0,
            train_subsets=["ll-10h"],
            val_subsets=["dev-clean"],
        )
        loader = dm.train_dataloader()
        total = sum(len(b) for b in loader.dataset.dataset.datasets[0].batches)
        self.assertEqual(total, 10)
        batch = next(iter(loader))
        text = decode_hubert_ltr(batch.targets[0][: batch.target_lengths[0]].tolist())
        self.assertTrue(text.startswith("HELLO WORLD"))
        self.assertEqual(len(dm.val_dataloader().dataset), 1)

class TestTeacherPipelineSmall(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        _fake_corpora(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_stream_writer_produces_standard_npy(self):

        rng = np.random.RandomState(0)
        parts = [rng.randn(n, 5).astype(np.float32) for n in (3, 7, 1)]
        path = self.root / "stream.npy"
        writer = NpyStreamWriter(path)
        for p in parts:
            writer.append(p)
        writer.close()
        self.assertTrue(np.array_equal(np.load(path), np.concatenate(parts)))
        self.assertEqual(np.load(path, mmap_mode="r").shape, (11, 5))

    def test_stream_writer_abort_removes_partial_file(self):

        path = self.root / "partial.npy"
        writer = NpyStreamWriter(path)
        writer.append(np.zeros((2, 3), dtype=np.float32))
        writer.abort()
        self.assertFalse(path.exists())

    def test_extract_mfcc_on_100h_splits_matches_direct_mfcc(self):

        out = self.root / "mfcc_100h"
        args = argparse.Namespace(
            librispeech_path=self.root, out_dir=out, feature_type="mfcc",
            checkpoint_path=None, model_size="base", num_classes="100", label_rate=50.0,
            layer=6, subsets=["train-clean-100", "dev-clean"], use_cuda=False,
            sanity_check=False,
        )
        run_extract(args)
        feats = np.load(out / "features.npy", mmap_mode="r")
        meta = json.loads((out / "index.json").read_text())
        self.assertEqual(sum(i["length"] for i in meta["index"]), feats.shape[0])
        self.assertEqual(feats.shape[1], 39)
        self.assertEqual([i["split"] for i in meta["index"]], ["train-clean-100"] * 3 + ["dev-clean"] * 3)
        first = open_subset(self.root, "train-clean-100")[0]
        direct = extract_mfcc_39(first[0].squeeze(0)).numpy()
        self.assertTrue(np.allclose(feats[: direct.shape[0]], direct, atol=1e-5))

    def test_default_fit_splits_drop_dev_and_test(self):

        index = [{"split": s, "length": 1} for s in ("train-clean-100", "dev-clean", "dev-other", "test-clean")]
        self.assertEqual(default_fit_splits(index), ["train-clean-100"])
        self.assertEqual(default_fit_splits([{"split": "dev-clean", "length": 1}]), ["dev-clean"])

    def test_split_row_ranges_merge_contiguous_blocks(self):

        index = [
            {"split": "a", "length": 3}, {"split": "a", "length": 2},
            {"split": "b", "length": 4}, {"split": "a", "length": 1},
        ]
        self.assertEqual(split_row_ranges(index, ["a"]), [(0, 5), (9, 10)])
        self.assertEqual(split_row_ranges(index, ["b"]), [(5, 9)])

    def test_sample_row_ranges_takes_only_requested_rows(self):

        feats = np.concatenate([np.zeros((50, 2)), np.full((20, 2), 99.0), np.ones((30, 2))]).astype(np.float32)
        rng = np.random.RandomState(0)
        out = sample_row_ranges(feats, [(0, 50), (70, 100)], percent=1.0, rng=rng)
        self.assertEqual(out.shape, (80, 2))
        self.assertFalse((out == 99.0).any())
        capped = sample_row_ranges(feats, [(0, 50), (70, 100)], percent=1.0, rng=rng, max_frames=40)
        self.assertEqual(capped.shape[0], 40)
        self.assertFalse((capped == 99.0).any())

class TestFinetunePresets(unittest.TestCase):
    def _args(self, preset, **over):
        base = dict(
            preset=preset, train_subsets=None, lr=None, max_steps=None, warmup_steps=None,
            hold_steps=None, decay_steps=None, ft_mask_prob=None, max_batch_duration=None,
            accumulate_grad_batches=None, gpus=4, nodes=1,
        )
        base.update(over)
        return argparse.Namespace(**base)

    def test_10h_preset_matches_fairseq_base_10h(self):

        args = apply_preset(self._args("10h"))
        self.assertEqual(args.train_subsets, ["ll-10h"])
        self.assertEqual(args.lr, 2e-5)
        self.assertEqual(args.max_steps, 25000)
        self.assertEqual((args.warmup_steps, args.hold_steps, args.decay_steps), (8000, 0, 72000))
        self.assertAlmostEqual(args.ft_mask_prob, 0.075)
        self.assertEqual(args.max_batch_duration * 4 * args.accumulate_grad_batches, 200.0)

    def test_100h_preset_keeps_1600s_batch(self):

        args = apply_preset(self._args("100h"))
        self.assertEqual(args.train_subsets, ["train-clean-100"])
        self.assertEqual((args.max_batch_duration, args.accumulate_grad_batches), (200.0, 2))

    def test_explicit_flags_override_preset(self):

        args = apply_preset(self._args("10h", lr=1e-4, train_subsets=["ll-1h"], max_batch_duration=40.0))
        self.assertEqual(args.lr, 1e-4)
        self.assertEqual(args.train_subsets, ["ll-1h"])
        self.assertEqual(args.max_batch_duration, 40.0)

    def test_10h_pt100h_schedule_fully_decays_within_max_steps(self):

        args = apply_preset(self._args("10h-pt100h"))
        self.assertEqual(args.train_subsets, ["ll-10h"])
        self.assertEqual(args.warmup_steps + args.hold_steps + args.decay_steps, args.max_steps)
        self.assertLess(args.freeze_steps, args.warmup_steps)
        self.assertEqual(args.max_batch_duration * 4 * args.accumulate_grad_batches, 200.0)

        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([param], lr=args.lr)
        sched = TriStageLRScheduler(
            opt, warmup_steps=args.warmup_steps, hold_steps=args.hold_steps,
            decay_steps=args.decay_steps, final_lr_scale=0.05,
        )
        self.assertAlmostEqual(sched._scale(args.max_steps - 1), 0.05, places=3)

    def test_random_init_disables_freeze_unless_explicit(self):

        args = apply_preset(self._args("10h", random_init=True))
        self.assertEqual(args.freeze_steps, 0)
        self.assertEqual(args.lr, 2e-5)
        args = apply_preset(self._args("10h", random_init=True, freeze_steps=500))
        self.assertEqual(args.freeze_steps, 500)
        args = apply_preset(self._args("10h"))
        self.assertEqual(args.freeze_steps, 10000)

    def test_cli_requires_explicit_init_choice(self):

        from experiments.hubert_train.finetune_hubert import run_train

        args = argparse.Namespace(random_init=False, pretrained_path=None, sanity_check=False)
        with self.assertRaisesRegex(ValueError, "--random-init"):
            run_train(args)
        args = argparse.Namespace(random_init=True, pretrained_path=Path("x.ckpt"), sanity_check=False)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            run_train(args)

class TestSmallConfig(unittest.TestCase):
    def test_small_is_narrower_and_shallower_than_base(self):

        small = HubertModel(get_hubert_config("small", num_classes=[100], label_rate=100.0))
        base = HubertModel(get_hubert_config("base", num_classes=[100], label_rate=100.0))
        n_small = sum(p.numel() for p in small.parameters())
        n_base = sum(p.numel() for p in base.parameters())
        self.assertEqual(small.cfg.encoder_embed_dim, 512)
        self.assertEqual(small.cfg.encoder_layers, 8)
        self.assertIsNone(small.post_extract_proj)
        self.assertLess(n_small, 0.4 * n_base)
        self.assertGreater(n_small, 25_000_000)

        source, lengths, targets = _tiny_batch(samples=16000, num_classes=100, label_rate=100.0)
        loss = small.masked_prediction_loss(small(source, lengths, targets, mask=True))
        self.assertTrue(torch.isfinite(loss))

class TestGer(unittest.TestCase):
    def _pretrain_args(self, **over):
        base = dict(
            model_size="tiny", num_classes="16", label_rate=50.0, mask_alpha=1.0, lr=1e-3,
            weight_decay=0.0, warmup_ratio=0.1, max_steps=None, ger_layer=None,
            ger_max_seconds=3600.0,
        )
        base.update(over)
        return argparse.Namespace(**base)

    def test_gram_rank_matches_matrix_rank(self):

        from experiments.hubert_train.compute_ger import effective_rank as cli_effective_rank
        from src.models.hubert.ger import effective_rank_from_gram

        x = torch.randn(500, 32) @ torch.diag(torch.linspace(0.1, 3.0, 32))
        gram = x.double().T @ x.double()
        self.assertAlmostEqual(effective_rank_from_gram(gram), cli_effective_rank(x), places=6)
        few = x[:10]
        self.assertAlmostEqual(
            effective_rank_from_gram(few.double().T @ few.double()), cli_effective_rank(few), places=4
        )

    def test_validation_accumulator_matches_offline_computation(self):

        from experiments.hubert_train.compute_ger import effective_rank as cli_effective_rank

        torch.manual_seed(0)
        module = HubertPretrainModule(self._pretrain_args())
        module.eval()
        self.assertEqual(module.ger_layer, module.model.cfg.encoder_layers)
        module._ger_reset()
        transform = DummyHubertPretrainTransform(num_classes=[16], label_rate=50.0)
        frames, sums = [], []
        for samples in _fake_librispeech_batches(4, 1):
            batch = transform(samples)
            module._ger_update(batch)
            with torch.no_grad():
                hidden, lengths, _ = module.model.extract_features(
                    batch.inputs, batch.input_lengths, tgt_layer=module.ger_layer - 1
                )
            h = hidden[0, : int(lengths[0])]
            frames.append(h)
            sums.append(h.sum(dim=0))

        from src.models.hubert.ger import effective_rank_from_gram

        self.assertAlmostEqual(
            effective_rank_from_gram(module._ger_gram), cli_effective_rank(torch.cat(frames)), places=5
        )
        self.assertAlmostEqual(
            effective_rank_from_gram(module._ger_utt_gram), cli_effective_rank(torch.stack(sums)), places=5
        )
        self.assertEqual(int(module._ger_counts[1]), 4)

    def test_budget_caps_audio(self):

        module = HubertPretrainModule(self._pretrain_args(ger_max_seconds=1.0))
        module.eval()
        module._ger_reset()
        transform = DummyHubertPretrainTransform(num_classes=[16], label_rate=50.0)
        for samples in _fake_librispeech_batches(6, 1, seconds=0.5):
            module._ger_update(transform(samples))
        self.assertAlmostEqual(float(module._ger_counts[2]), 1.0)

    def test_ger_logged_at_every_pretrain_epoch_end(self):

        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import CSVLogger
        from src.data.librispeech_data_module import TransformDataset

        torch.manual_seed(0)
        transform = DummyHubertPretrainTransform(num_classes=[16], label_rate=50.0)
        train = torch.utils.data.DataLoader(
            TransformDataset(_fake_librispeech_batches(3, 2, seed=1), transform), batch_size=None
        )
        val = torch.utils.data.DataLoader(
            TransformDataset(_fake_librispeech_batches(2, 2, seed=2), transform), batch_size=None
        )
        module = HubertPretrainModule(self._pretrain_args())
        with tempfile.TemporaryDirectory() as tmp:
            logger = CSVLogger(tmp, name="lightning_logs")
            trainer = Trainer(
                max_epochs=3, logger=logger, accelerator="cpu", devices=1,
                enable_checkpointing=False, enable_progress_bar=False, num_sanity_val_steps=1,
                enable_model_summary=False,
            )
            trainer.fit(module, train, val)
            with open(Path(logger.log_dir) / "metrics.csv", encoding="utf-8") as handle:
                rows = [r for r in csv.DictReader(handle) if r.get("Metrics/val_ger")]

        self.assertEqual([int(r["epoch"]) for r in rows], [0, 1, 2])
        self.assertEqual([int(r["step"]) for r in rows], [2, 5, 8])
        for r in rows:
            self.assertGreater(float(r["Metrics/val_ger"]), 1.0)
            self.assertLessEqual(float(r["Metrics/val_ger"]), module.model.cfg.encoder_embed_dim)
            self.assertTrue(r["Metrics/val_rankme_t"])
            self.assertTrue(r["Losses/val_loss"])

    def test_ger_disabled_with_zero_budget(self):

        module = HubertPretrainModule(self._pretrain_args(ger_max_seconds=0.0))
        module._ger_update(None)
        self.assertFalse(hasattr(module, "_ger_gram"))

class TestFinetuneFit(unittest.TestCase):
    def test_random_init_finetune_fit_trains_cnn_and_transformer(self):

        from pytorch_lightning import Trainer
        from src.data.librispeech_data_module import TransformDataset

        torch.manual_seed(0)
        args = argparse.Namespace(
            model_size="tiny", num_classes="16", label_rate=50.0, mask_alpha=1.0, lr=1e-3,
            freeze_steps=0, unfreeze_steps=0, warmup_steps=2, hold_steps=2, decay_steps=2,
            pretrained_path=None, random_init=True, apply_ft_mask=False, layerdrop=0.0,
        )
        module = HubertCTCModule(args)
        before = {k: v.clone() for k, v in module.encoder.state_dict().items()}
        transform = HubertCharFinetuneTransform()
        loader = torch.utils.data.DataLoader(
            TransformDataset(_fake_librispeech_batches(3, 2, seconds=1.0), transform), batch_size=None
        )
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                max_steps=3, logger=False, accelerator="cpu", devices=1, default_root_dir=tmp,
                enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                limit_val_batches=0,
            )
            trainer.fit(module, loader)
        after = module.encoder.state_dict()
        cnn_changed = [
            not torch.equal(before[k], after[k]) for k in after if k.startswith("feature_extractor") and "weight" in k
        ]
        layer_changed = [
            not torch.equal(before[k], after[k]) for k in after if k.startswith("encoder.layers.") and k.endswith("fc1.weight")
        ]
        self.assertTrue(any(cnn_changed))
        self.assertTrue(all(layer_changed))

class TestScheduler(unittest.TestCase):
    def test_tri_stage_warmup_hold_then_exponential_decay(self):

        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([param], lr=1.0)
        sched = TriStageLRScheduler(
            opt, warmup_steps=10, hold_steps=40, decay_steps=50, final_lr_scale=0.05
        )
        lrs = []
        for _ in range(100):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()

        self.assertAlmostEqual(lrs[0], 0.01, places=6)
        self.assertLess(lrs[0], lrs[5])
        self.assertAlmostEqual(lrs[10], 1.0, places=6)
        self.assertAlmostEqual(lrs[49], 1.0, places=6)
        self.assertLess(lrs[60], lrs[50])
        self.assertAlmostEqual(sched._scale(100), 0.05, places=6)
        self.assertAlmostEqual(sched._scale(500), 0.05, places=6)

    def test_linear_warmup_then_decay(self):

        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([param], lr=1.0)
        sched = LinearWarmupDecayScheduler(opt, warmup_steps=10, total_steps=100)
        lrs = []
        for _ in range(100):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        self.assertGreater(lrs[4], lrs[0])
        peak = max(lrs)
        self.assertGreater(peak, 0.9)
        self.assertLess(lrs[-1], lrs[20])
        self.assertLess(lrs[-1], 0.15)

def _has_module(name):
    import importlib.util

    return importlib.util.find_spec(name) is not None

@unittest.skipUnless(_has_module("umap") and _has_module("matplotlib"), "umap-learn / matplotlib not installed")
class TestUmapEmbeddings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        _fake_corpora(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _args(self, out_dir, **over):
        base = dict(
            checkpoint_path=None, random_init=False, librispeech_path=self.root, out_dir=out_dir,
            subsets=["ll-10h"], model_size=None, num_classes=None, label_rate=None, layer=None,
            num_speakers=20, utts_per_speaker=10, frames_per_utt=20, km_path=None, km_labels=None,
            km_label_rate=100.0, phone_alignments=None, phone_rate=100.0, phone_mapping=None,
            letter_source="align", n_clusters=4, n_neighbors=5, min_dist=0.1, metric="cosine",
            knn=3, point_size=2.0, label=None, save_embeddings=False, seed=0, use_cuda=False,
        )
        base.update(over)
        return argparse.Namespace(**base)

    def test_finetuned_checkpoint_plots_every_colouring(self):

        import joblib

        from experiments.hubert_train.umap_embeddings import run_umap

        torch.manual_seed(0)
        module = HubertCTCModule(argparse.Namespace(
            model_size="tiny", num_classes="16", label_rate=50.0, mask_alpha=1.0, lr=1e-3,
            freeze_steps=0, warmup_steps=1, pretrained_path=None,
        ))
        out = self.root / "umap_ft"
        ckpt = self.root / "ft.ckpt"
        torch.save({"state_dict": module.state_dict(), "hyper_parameters": {"model_size": "tiny",
                    "num_classes": "16", "label_rate": 50.0}, "global_step": 7}, ckpt)
        km = MiniBatchKMeans(n_clusters=5, random_state=0, n_init=1).fit(np.random.RandomState(0).randn(200, 39))
        km_path = self.root / "km.bin"
        joblib.dump(km, km_path)
        phones = self.root / "phones.txt"
        phones.write_text("103-1241-0000 " + " ".join(["3"] * 100) + "\n103-1241-0001 1 2 3\n")

        meta = run_umap(self._args(out, checkpoint_path=ckpt, km_path=km_path, phone_alignments=phones))

        self.assertEqual(meta["model_size"], "tiny")
        self.assertEqual(meta["checkpoint_step"], 7)
        self.assertTrue(meta["finetuned"])
        self.assertEqual(meta["n_speakers"], 2)
        self.assertEqual(meta["n_utterances"], 10)
        self.assertEqual(meta["stats"]["phones_mismatched"], 1)
        for key in ("speaker", "letter", "teacher", "emb_kmeans", "utterance_speaker"):
            self.assertIn(key, meta["purity"])
        for name in ("speaker", "letter", "teacher", "emb_kmeans", "phone"):
            self.assertTrue((out / f"umap_frames_{name}.png").is_file(), name)
        self.assertTrue((out / "umap_utterances_speaker.png").is_file())
        self.assertTrue((out / "umap_overview.png").is_file())
        points = np.load(out / "umap_points.npz")
        self.assertEqual(points["coords"].shape, (meta["n_frames"], 2))
        self.assertEqual(points["letter"].shape[0], meta["n_frames"])
        self.assertTrue((points["letter"] >= 0).any())
        self.assertNotIn("frames", points.files)

    def test_random_init_reference_needs_no_checkpoint(self):

        from experiments.hubert_train.umap_embeddings import run_umap

        out = self.root / "umap_rand"
        meta = run_umap(self._args(out, random_init=True, model_size="tiny", num_classes="16", label_rate=50.0))
        self.assertEqual(meta["init"], "random")
        self.assertNotIn("letter", meta["purity"])
        self.assertTrue((out / "umap_frames_speaker.png").is_file())
        self.assertFalse((out / "umap_frames_letter.png").exists())

if __name__ == "__main__":
    unittest.main()
