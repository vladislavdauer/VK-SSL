import itertools

import torch
from pytorch_lightning import LightningModule
from torchmetrics.text import WordErrorRate

from src.models.hubert.config import get_hubert_config
from src.models.hubert.hubert_model import HubertModel
from src.data.hubert_transforms import HUBERT_BLANK_ID, decode_hubert_ltr
from src.opt.schedulers import LinearWarmupDecayScheduler, TriStageLRScheduler

_expected_spm_vocab_size = 128


def _hubert_model_kwargs(args):
    return dict(
        num_classes=[int(v) for v in str(args.num_classes).split(",") if str(v).strip()],
        label_rate=float(args.label_rate),
        mask_alpha=float(args.mask_alpha),
        mask_prob=float(getattr(args, "mask_prob", 0.80)),
        feature_grad_mult=float(getattr(args, "feature_grad_mult", 0.1)),
        fairseq_mask=bool(getattr(args, "fairseq_mask", True)),
    )


class HubertPretrainModule(LightningModule):
    def __init__(self, args=None):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.model = HubertModel(
            get_hubert_config(
                args.model_size,
                **_hubert_model_kwargs(args),
            )
        )
        self.optimizer = torch.optim.Adam(
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
        self.label_type = str(getattr(args, "label_type", "char")).lower()
        if self.label_type == "spm":
            spm_vocab_size = self.sp_model.get_piece_size()
            assert spm_vocab_size == _expected_spm_vocab_size, (
                f"SPM vocab size ({spm_vocab_size}) must be equal to expected vocab size ({_expected_spm_vocab_size})"
            )
            self.blank_idx = spm_vocab_size
            ctc_vocab_size = spm_vocab_size + 1
        else:
            self.blank_idx = HUBERT_BLANK_ID
            ctc_vocab_size = HUBERT_BLANK_ID + 1

        self.encoder = HubertModel(
            get_hubert_config(args.model_size, **_hubert_model_kwargs(args))
        )
        if getattr(args, "pretrained_path", None):
            self._load_pretrained(args.pretrained_path)

        self.encoder.remove_pretraining_modules()
        self.encoder.freeze_feature_extractor()
        self.encoder.encoder.layerdrop = float(getattr(args, "layerdrop", 0.1))
        self.encoder.encoder.dropout = 0.0
        for layer in self.encoder.encoder.layers:
            layer.dropout = 0.0
            layer.activation_dropout = 0.1
            layer.self_attn.dropout = 0.0
        self.apply_ft_mask = bool(getattr(args, "apply_ft_mask", True))
        self.ft_mask_prob = float(getattr(args, "ft_mask_prob", 0.75))
        self.ft_mask_channel_prob = float(getattr(args, "ft_mask_channel_prob", 0.5))
        self.ft_mask_channel_length = int(getattr(args, "ft_mask_channel_length", 64))

        self.ctc_out = torch.nn.Linear(self.encoder.cfg.encoder_embed_dim, ctc_vocab_size)
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
            lr=float(getattr(args, "lr", 2e-5)),
            eps=float(getattr(args, "adam_eps", 1e-8)),
            betas=(0.9, 0.98),
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

    def _greedy_texts(self, log_probs, src_lengths, batch):
        pred_ids = log_probs.argmax(dim=-1)

        pred_texts = []
        target_texts = []

        for i in range(pred_ids.size(0)):
            src_len = int(src_lengths[i].item())
            seq = pred_ids[i, :src_len]

            collapsed = []
            prev = self.blank_idx

            for idx in seq:
                idx = int(idx.item())

                if idx != prev and idx != self.blank_idx:
                    collapsed.append(idx)

                prev = idx

            pred_texts.append(
                self.sp_model.decode(collapsed)
                if self.label_type == "spm"
                else decode_hubert_ltr(collapsed)
            )

            target_len = int(batch.target_lengths[i].item())
            target_ids = (
                batch.targets[i, :target_len]
                .detach()
                .cpu()
                .tolist()
            )
            if self.label_type == "spm":
                target_texts.append(self.sp_model.decode(target_ids))
            else:
                target_texts.append(decode_hubert_ltr(target_ids))

        return pred_texts, target_texts

    def _step(self, batch, step_type):
        if batch is None:
            return None

        encoded, src_lengths, _ = self.encoder.extract_features(
            batch.inputs,
            batch.input_lengths,
            mask=self.apply_ft_mask and self.training,
            mask_prob=self.ft_mask_prob,
            fairseq_style=True,
            mask_channel_prob=(
                self.ft_mask_channel_prob if (self.apply_ft_mask and self.training) else 0.0
            ),
            mask_channel_length=self.ft_mask_channel_length,
        )

        logits = self.ctc_out(encoded)
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

                if step_type == "train":
                    self.train_wer.update(pred_texts, target_texts)
                    self.log(
                        "Metrics/train_wer",
                        self.train_wer,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                    )

                if step_type == "val":
                    self.val_wer.update(pred_texts, target_texts)
                    self.log(
                        "Metrics/val_wer",
                        self.val_wer,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                    )

                if step_type == "test":
                    self.test_wer.update(pred_texts, target_texts)
                    self.log(
                        "Metrics/test_wer",
                        self.test_wer,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                    )

        return loss

    def forward(self, batch):
        features = batch.inputs.to(self.device)
        lengths = batch.input_lengths.to(self.device)
        encoded, src_lengths, _ = self.encoder.extract_features(features, lengths)
        logits = self.ctc_out(encoded)
        log_probs = self.log_softmax(logits)

        predicted_ids = torch.argmax(log_probs, dim=-1)

        results = []
        for seq, length in zip(predicted_ids, src_lengths):
            seq = seq[: int(length.item())]

            collapsed = []
            prev = self.blank_idx

            for idx in seq:
                idx = int(idx.item())
                if idx != prev and idx != self.blank_idx:
                    collapsed.append(idx)
                prev = idx

            results.append(
                self.sp_model.decode(collapsed)
                if self.label_type == "spm"
                else decode_hubert_ltr(collapsed)
            )

        return results[0] if len(results) == 1 else results

    def configure_optimizers(self):
        decay_steps = int(getattr(self.args, "decay_steps", 17000))
        self.lr_scheduler = TriStageLRScheduler(
            self.optimizer,
            init_lr=float(getattr(self.args, "lr", 2e-5)),
            warmup_steps=int(getattr(self.args, "warmup_steps", 8000)),
            hold_steps=int(getattr(self.args, "hold_steps", 0)),
            decay_steps=decay_steps,
            final_lr_scale=float(getattr(self.args, "final_lr_scale", 0.05)),
        )

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {"scheduler": self.lr_scheduler, "interval": "step"},
        }

    def on_before_optimizer_step(self, optimizer):
        encoder_grad_norm = torch.zeros((), device=self.device)
        head_grad_norm = torch.zeros((), device=self.device)

        for parameter in self.encoder.parameters():
            if parameter.grad is not None:
                encoder_grad_norm += parameter.grad.detach().float().pow(2).sum()

        for parameter in self.ctc_out.parameters():
            if parameter.grad is not None:
                head_grad_norm += parameter.grad.detach().float().pow(2).sum()

        self.log(
            "Norms/encoder_grad_norm",
            encoder_grad_norm.sqrt(),
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "Norms/head_grad_norm",
            head_grad_norm.sqrt(),
            on_step=True,
            on_epoch=False,
        )

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")

        encoder_weight_norm = torch.zeros((), device=self.device)
        head_weight_norm = torch.zeros((), device=self.device)

        for parameter in self.encoder.parameters():
            encoder_weight_norm += parameter.detach().float().pow(2).sum()

        for parameter in self.ctc_out.parameters():
            head_weight_norm += parameter.detach().float().pow(2).sum()

        self.log(
            "Norms/encoder_weight_norm",
            encoder_weight_norm.sqrt(),
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "Norms/head_weight_norm",
            head_weight_norm.sqrt(),
            on_step=True,
            on_epoch=False,
        )

        self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")
