import argparse
import json
import pickle
import tempfile
import unittest
from pathlib import Path

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
        encoder_trainable = [
            p.requires_grad
            for n, p in module.encoder.named_parameters()
            if not n.startswith("feature_extractor")
        ]
        self.assertTrue(encoder_trainable)
        self.assertFalse(any(encoder_trainable))

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

if __name__ == "__main__":
    unittest.main()
