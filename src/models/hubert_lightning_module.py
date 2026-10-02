import itertools

import torch
from pytorch_lightning import LightningModule
from torchmetrics.text import WordErrorRate

from src.data.hubert_transforms import HUBERT_LTR_VOCAB, decode_hubert_ltr
from src.models.hubert.config import get_hubert_config
from src.models.hubert.hubert_model import HubertModel
from src.opt.schedulers import LinearWarmupDecayScheduler, TriStageLRScheduler

_expected_spm_vocab_size = 128

class HubertPretrainModule(LightningModule):
    def __init__(self, args=None):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        num_classes = [int(v) for v in str(args.num_classes).split(",") if str(v).strip()]
        self.model = HubertModel(
            get_hubert_config(
                args.model_size,
                num_classes=num_classes,
                label_rate=float(args.label_rate),
                mask_alpha=float(args.mask_alpha),
                mask_prob=getattr(args, "mask_prob", None),
            )
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(args.lr),
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=float(args.weight_decay),
        )

    def _step(self, batch, step_type):
        if batch is None:
            return None

        net_output = self.model(
            source=batch.inputs,
            lengths=batch.input_lengths,
            target_list=list(batch.targets),
            mask=True,
        )
        loss = self.model.masked_prediction_loss(net_output)
        self.log(f"Losses/{step_type}_loss", loss, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log(
            f"Losses/{step_type}_features_pen",
            net_output["features_pen"],
            on_epoch=True,
            sync_dist=True,
        )

        masked = net_output["mask_indices"]
        valid = ~net_output["padding_mask"]
        self.log(
            f"Metrics/{step_type}_mask_ratio",
            (masked & valid).float().sum() / valid.float().sum().clamp_min(1.0),
            on_step=True,
            on_epoch=False,
        )
        if net_output["logit_m_list"] and net_output["logit_m_list"][0].numel() > 0:
            acc = (
                net_output["logit_m_list"][0].argmax(dim=-1)
                == net_output["target_m_list"][0]
            ).float().mean()
            self.log(f"Metrics/{step_type}_masked_acc", acc, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")

        self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        if getattr(self.args, "max_steps", None):
            total_steps = int(self.args.max_steps)
        else:
            total_steps = int(self.trainer.estimated_stepping_batches)

        warmup_steps = int(float(self.args.warmup_ratio) * total_steps)
        scheduler = LinearWarmupDecayScheduler(
            self.optimizer,
            warmup_steps=max(warmup_steps, 1),
            total_steps=max(total_steps, 2),
        )

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

class HubertCTCModule(LightningModule):
    def __init__(self, args=None, sp_model=None):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.sp_model = sp_model
        self.label_type = getattr(args, "label_type", "char")
        if self.label_type == "spm":
            vocab_size = self.sp_model.get_piece_size()
            assert vocab_size == _expected_spm_vocab_size, (
                f"SPM vocab size ({vocab_size}) must be equal to expected vocab size ({_expected_spm_vocab_size})"
            )
        else:
            vocab_size = len(HUBERT_LTR_VOCAB)

        self.blank_idx = vocab_size

        num_classes = [int(v) for v in str(args.num_classes).split(",") if str(v).strip()]
        self.encoder = HubertModel(
            get_hubert_config(
                args.model_size,
                num_classes=num_classes,
                label_rate=float(args.label_rate),
                mask_alpha=float(args.mask_alpha),
            )
        )
        if getattr(args, "pretrained_path", None):
            self._load_pretrained(args.pretrained_path)

        self.encoder.remove_pretraining_modules()
        self.encoder.freeze_feature_extractor()
        self.encoder.set_finetune_dropout(
            dropout=float(getattr(args, "dropout", 0.0)),
            attention_dropout=float(getattr(args, "attention_dropout", 0.0)),
            activation_dropout=float(getattr(args, "activation_dropout", 0.1)),
            layerdrop=float(getattr(args, "layerdrop", 0.1)),
        )

        self.final_dropout = torch.nn.Dropout(float(getattr(args, "final_dropout", 0.0)))
        self.ctc_out = torch.nn.Linear(self.encoder.cfg.encoder_embed_dim, vocab_size + 1)
        torch.nn.init.normal_(self.ctc_out.weight, mean=0.0, std=0.01)
        torch.nn.init.constant_(self.ctc_out.bias, 0.0)

        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

        self.loss = torch.nn.CTCLoss(blank=self.blank_idx, reduction="none", zero_infinity=True)

        trainable = itertools.chain(
            *[
                [p for p in self.encoder.parameters() if p.requires_grad],
                self.ctc_out.parameters(),
            ]
        )
        self.optimizer = torch.optim.Adam(
            trainable,
            lr=float(args.lr),
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        freeze_steps = int(getattr(args, "freeze_steps", 0))
        if freeze_steps > 0:
            self._set_transformer_frozen(True)
        else:
            self._frozen_transformer = False

        self.train_wer = WordErrorRate()
        self.val_wer = WordErrorRate()
        self.test_wer = WordErrorRate()

    def _load_pretrained(self, path):
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        cleaned = {}
        for key, value in state.items():
            if key.startswith("model."):
                cleaned[key[len("model.") :]] = value
            elif key.startswith("encoder."):
                cleaned[key[len("encoder.") :]] = value
            else:
                cleaned[key] = value

        missing, unexpected = self.encoder.load_state_dict(cleaned, strict=False)
        _ = missing, unexpected

    def _set_transformer_frozen(self, frozen: bool):
        for name, parameter in self.encoder.named_parameters():
            if name.startswith("feature_extractor"):
                parameter.requires_grad = False
            else:
                parameter.requires_grad = not frozen

        self._frozen_transformer = frozen

    def on_train_batch_start(self, batch, batch_idx):
        freeze_steps = int(getattr(self.args, "freeze_steps", 0))
        should_freeze = self.global_step < freeze_steps
        if should_freeze != self._frozen_transformer:
            self._set_transformer_frozen(should_freeze)

    def _decode_ids(self, token_ids):
        if self.label_type == "spm":
            return self.sp_model.decode(list(token_ids))

        return decode_hubert_ltr(token_ids)

    def _collapse(self, seq):
        collapsed = []
        prev = self.blank_idx
        for idx in seq.tolist():
            idx = int(idx)
            if idx != prev and idx != self.blank_idx:
                collapsed.append(idx)

            prev = idx

        return collapsed

    def _greedy_texts(self, log_probs, src_lengths, batch):
        pred_ids = log_probs.argmax(dim=-1)

        pred_texts = []
        target_texts = []

        for i in range(pred_ids.size(0)):
            src_len = int(src_lengths[i].item())
            pred_texts.append(self._decode_ids(self._collapse(pred_ids[i, :src_len])))

            target_len = int(batch.target_lengths[i].item())
            target_ids = batch.targets[i, :target_len].detach().cpu().tolist()
            target_texts.append(self._decode_ids(target_ids))

        return pred_texts, target_texts

    def _step(self, batch, step_type):
        if batch is None:
            return None

        apply_mask = bool(getattr(self.args, "apply_ft_mask", True)) and step_type == "train"
        encoded, src_lengths, _ = self.encoder.extract_features(
            batch.inputs,
            batch.input_lengths,
            mask=apply_mask,
            mask_prob=float(getattr(self.args, "ft_mask_prob", 0.065)),
            mask_channel_prob=(
                float(getattr(self.args, "ft_mask_channel_prob", 0.5)) if apply_mask else 0.0
            ),
            mask_channel_length=int(getattr(self.args, "ft_mask_channel_length", 64)),
        )

        logits = self.ctc_out(self.final_dropout(encoded))
        probs = self.log_softmax(logits).transpose(0, 1)

        loss = self.loss(
            probs,
            batch.targets,
            src_lengths,
            batch.target_lengths,
        ).mean()

        self.log(f"Losses/{step_type}_loss", loss, on_epoch=True, sync_dist=True)

        if step_type in ("train", "val", "test"):
            with torch.no_grad():
                pred_texts, target_texts = self._greedy_texts(
                    probs.transpose(0, 1), src_lengths, batch
                )
                metric = {
                    "train": self.train_wer,
                    "val": self.val_wer,
                    "test": self.test_wer,
                }[step_type]
                metric.update(pred_texts, target_texts)
                self.log(
                    f"Metrics/{step_type}_wer",
                    metric,
                    on_step=step_type == "train",
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

        return loss

    def forward(self, batch):
        features = batch.inputs.to(self.device)
        lengths = batch.input_lengths.to(self.device)
        encoded, src_lengths, _ = self.encoder.extract_features(features, lengths)
        logits = self.ctc_out(self.final_dropout(encoded))
        log_probs = self.log_softmax(logits)

        predicted_ids = torch.argmax(log_probs, dim=-1)

        results = [
            self._decode_ids(self._collapse(seq[: int(length.item())]))
            for seq, length in zip(predicted_ids, src_lengths)
        ]

        return results[0] if len(results) == 1 else results

    def configure_optimizers(self):
        scheduler = TriStageLRScheduler(
            self.optimizer,
            warmup_steps=int(getattr(self.args, "warmup_steps", 8000)),
            hold_steps=int(getattr(self.args, "hold_steps", 32000)),
            decay_steps=int(getattr(self.args, "decay_steps", 40000)),
            final_lr_scale=float(getattr(self.args, "final_lr_scale", 0.05)),
        )

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")

        self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")
