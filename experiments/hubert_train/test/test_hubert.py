import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data.hubert_transforms import (
    DummyHubertPretrainTransform,
    load_km_file,
    librispeech_utt_id,
)
from src.models.hubert.config import get_hubert_config, hubert_base, hubert_tiny
from src.models.hubert.hubert_model import HubertModel
from src.models.hubert.kmeans import (
    extract_mfcc_39,
    fit_kmeans,
    predict_labels,
    split_labels,
)
from src.models.hubert.masking import apply_mask, compute_span_mask
from src.models.hubert.prediction_head import HubertPredictionHead
from src.models.hubert_lightning_module import HubertCTCModule, HubertPretrainModule
from src.opt.schedulers import LinearWarmupDecayScheduler


def _tiny_batch(batch_size=2, samples=3200, num_classes=16, label_rate=50.0):
    source = torch.randn(batch_size, samples)
    lengths = torch.tensor([samples, samples // 2], dtype=torch.long)
    hop = int(16000 / label_rate)
    n_lab = max(samples // hop, 8)
    targets = torch.randint(0, num_classes, (batch_size, n_lab))
    return source, lengths, [targets]


class TestCnnEncoder(unittest.TestCase):
    def test_output_length_matches_stacked_convs(self):
        # считает длину после 7 conv слоёв тем же правилом, что и энкодер
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
        # проверяет, что T на выходе CNN совпадает с output_lengths
        model = HubertModel(hubert_tiny())
        source = torch.randn(2, 16000)
        lengths = torch.tensor([16000, 12000], dtype=torch.long)
        feats = model.feature_extractor(source)
        feat_lengths = model.feature_extractor.output_lengths(lengths)
        self.assertEqual(feats.shape[2], int(feat_lengths.max().item()))
        self.assertEqual(feats.shape[1], 32)


class TestMasking(unittest.TestCase):
    def test_mask_stays_inside_valid_frames(self):
        # маска не заходит на padding
        lengths = torch.tensor([40, 11, 3], dtype=torch.long)
        mask = compute_span_mask(lengths, mask_prob=0.08, mask_length=10, min_masks=2)
        for i, length in enumerate(lengths.tolist()):
            self.assertFalse(mask[i, length:].any())
            if length > 1:
                self.assertTrue(mask[i, :length].any())

    def test_mask_uses_span_length(self):
        # старты маски покрывают непрерывные span-ы длины l
        lengths = torch.tensor([80], dtype=torch.long)
        mask = compute_span_mask(lengths, mask_prob=0.08, mask_length=10, min_masks=2)
        idx = torch.where(mask[0])[0].tolist()
        self.assertGreaterEqual(len(idx), 10)

    def test_apply_mask_writes_embedding(self):
        # masked позиции заменяются обучаемым mask embedding
        x = torch.zeros(1, 5, 4)
        mask = torch.tensor([[False, True, True, False, False]])
        emb = torch.ones(4)
        y = apply_mask(x, mask, emb)
        self.assertTrue(torch.allclose(y[0, 1], emb))
        self.assertTrue(torch.allclose(y[0, 0], torch.zeros(4)))


class TestPredictionHead(unittest.TestCase):
    def test_logits_shape_and_temperature(self):
        # cosine-softmax логиты имеют форму (N, C) и делятся на tau
        head = HubertPredictionHead(embed_dim=8, final_dim=4, num_classes=[10], logit_temp=0.1)
        hidden = torch.randn(6, 8)
        logits = head.logits(hidden, 0)
        self.assertEqual(tuple(logits.shape), (6, 10))

    def test_ensemble_returns_one_logit_tensor_per_codebook(self):
        # ансамбль кластеров даёт отдельную голову на каждый codebook
        head = HubertPredictionHead(embed_dim=8, final_dim=4, num_classes=[5, 7])
        hidden = torch.randn(3, 8)
        outs = head(hidden)
        self.assertEqual(len(outs), 2)
        self.assertEqual(tuple(outs[0].shape), (3, 5))
        self.assertEqual(tuple(outs[1].shape), (3, 7))


class TestHubertMaskedLoss(unittest.TestCase):
    def test_alpha_one_matches_masked_cross_entropy(self):
        # при alpha=1 loss считается только по masked кадрам
        cfg = hubert_tiny(num_classes=[16], label_rate=50.0, mask_alpha=1.0)
        model = HubertModel(cfg)
        source, lengths, targets = _tiny_batch()
        torch.manual_seed(0)
        out = model(source, lengths, targets, mask=True)
        loss = model.masked_prediction_loss(out)
        manual = torch.nn.functional.cross_entropy(
            out["logit_m_list"][0].float(), out["target_m_list"][0].long()
        )
        self.assertTrue(torch.allclose(loss, manual, atol=1e-5))

    def test_alpha_zero_matches_unmasked_cross_entropy(self):
        # при alpha=0 модель учится только на видимых кадрах
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
        self.assertTrue(torch.allclose(loss, manual, atol=1e-5))

    def test_padding_excluded_from_masked_targets(self):
        # padded кадры не попадают в masked/unmasked loss
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
        # loss по нескольким k-means учителям усредняется
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
        self.assertTrue(torch.allclose(loss, torch.stack(parts).mean(), atol=1e-5))


class TestAlignAndLayers(unittest.TestCase):
    def test_label_alignment_subsamples_100hz_to_cnn_rate(self):
        # label_rate=100 выравнивает MFCC 10ms метки к кадрам CNN 20ms
        cfg = hubert_tiny(num_classes=[16], label_rate=100.0)
        model = HubertModel(cfg)
        self.assertAlmostEqual(model.feat2tar_ratio, 2.0)
        features = torch.randn(1, 32, 10)
        labels = torch.arange(20).view(1, 20)
        _, aligned = model.align_targets(features, [labels])
        self.assertEqual(aligned[0].shape[-1], 10)
        self.assertEqual(aligned[0][0, 1].item(), 2)

    def test_extract_features_returns_requested_layer(self):
        # extract_features(tgt_layer) возвращает выход выбранного transformer слоя
        model = HubertModel(hubert_tiny())
        source = torch.randn(1, 3200)
        lengths = torch.tensor([3200], dtype=torch.long)
        x0, _, _ = model.extract_features(source, lengths, tgt_layer=0)
        x1, _, _ = model.extract_features(source, lengths, tgt_layer=1)
        self.assertEqual(x0.shape, x1.shape)
        self.assertFalse(torch.allclose(x0, x1))


class TestKmeansAndMfcc(unittest.TestCase):
    def test_mfcc_has_39_dimensions(self):
        # MFCC + delta + delta-delta дают 39 признаков как в статье
        wav = torch.sin(torch.linspace(0, 200, 16000))
        feats = extract_mfcc_39(wav, 16000)
        self.assertEqual(feats.shape[1], 39)
        self.assertGreater(feats.shape[0], 10)

    def test_kmeans_fit_and_predict_roundtrip(self):
        # MiniBatchKMeans учится на точках и возвращает id кластеров
        rng = np.random.RandomState(0)
        a = rng.randn(80, 4) + np.array([5.0, 0, 0, 0])
        b = rng.randn(80, 4) + np.array([-5.0, 0, 0, 0])
        feats = np.concatenate([a, b], axis=0).astype(np.float32)
        km = fit_kmeans(feats, n_clusters=2, percent=1.0, n_init=1, max_iter=20, seed=0)
        labels = predict_labels(km, feats)
        self.assertEqual(labels.shape[0], 160)
        self.assertEqual(len(set(labels.tolist())), 2)

    def test_split_labels_restores_utterance_boundaries(self):
        # split_labels режет общий массив меток обратно по utterance
        split = split_labels([1, 2, 3, 4, 5], [2, 3])
        self.assertEqual(split, [[1, 2], [3, 4, 5]])

    def test_load_km_file_parses_utt_and_codes(self):
        # .km файл читается как словарь utterance -> последовательность кластеров
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.km"
            path.write_text("19-198-0001 0 1 1 2\n19-198-0002 3 3\n", encoding="utf-8")
            loaded = load_km_file(str(path))
        self.assertEqual(loaded["19-198-0001"], [0, 1, 1, 2])
        self.assertEqual(loaded["19-198-0002"], [3, 3])


class TestConfigAndTransforms(unittest.TestCase):
    def test_base_config_matches_paper_table(self):
        # BASE совпадает с Table I: 12 слоёв, 768 dim, 8 голов, proj 256
        cfg = hubert_base(num_classes=[100])
        self.assertEqual(cfg.encoder_layers, 12)
        self.assertEqual(cfg.encoder_embed_dim, 768)
        self.assertEqual(cfg.encoder_ffn_embed_dim, 3072)
        self.assertEqual(cfg.encoder_attention_heads, 8)
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

    def test_get_hubert_config_rejects_unknown_name(self):
        # неизвестный размер модели должен падать с ValueError
        with self.assertRaises(ValueError):
            get_hubert_config("huge")

    def test_dummy_pretrain_transform_builds_padded_batch(self):
        # dummy-метки позволяют прогнать collate без готового k-means
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
        # id реплики совпадает с файлами LibriSpeech
        sample = (torch.zeros(1, 10), 16000, "hi", 19, 198, 1)
        self.assertEqual(librispeech_utt_id(sample), "19-198-0001")


class TestTrainingStep(unittest.TestCase):
    def test_tiny_forward_backward(self):
        # полный forward+backward tiny HuBERT не даёт nan и пишет градиенты
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
        # Lightning pretrain step считает masked prediction loss
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
        # CNN замораживается для CTC fine-tune, как в статье
        model = HubertModel(hubert_tiny())
        model.freeze_feature_extractor()
        for p in model.feature_extractor.parameters():
            self.assertFalse(p.requires_grad)
        self.assertEqual(model.feature_grad_mult, 0.0)

    def test_remove_pretraining_modules_drops_head(self):
        # projection/codebook снимаются перед CTC головой
        model = HubertModel(hubert_tiny())
        model.remove_pretraining_modules()
        self.assertIsNone(model.pred_head)

    def test_ctc_module_greedy_decode_and_cnn_frozen(self):
        # CTC fine-tune держит CNN frozen и декодирует greedy без blank/repeat
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
        # freeze_steps оставляет transformer выключенным на первых шагах fine-tune
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


class TestScheduler(unittest.TestCase):
    def test_linear_warmup_then_decay(self):
        # lr линейно греется 8% шагов и линейно падает к нулю
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
