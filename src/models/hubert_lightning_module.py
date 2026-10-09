import contextlib

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_info
from torchmetrics.text import WordErrorRate

from src.data.hubert_transforms import HUBERT_LTR_VOCAB, decode_hubert_ltr
from src.models.hubert.config import get_hubert_config
from src.models.hubert.ger import effective_rank_from_gram
from src.models.hubert.hubert_model import HubertModel, load_encoder_state
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
        self.ger_layer = int(getattr(args, "ger_layer", None) or self.model.cfg.encoder_layers)
        self.ger_max_seconds = float(getattr(args, "ger_max_seconds", 0.0) or 0.0)
        if not 1 <= self.ger_layer <= self.model.cfg.encoder_layers:
            raise ValueError(
                f"--ger-layer must be in [1, {self.model.cfg.encoder_layers}], got {self.ger_layer}"
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
        loss = self._step(batch, "val")
        self._ger_update(batch)
        return loss

    def _ger_reset(self):
        dim = self.model.cfg.encoder_embed_dim
        self._ger_gram = torch.zeros(dim, dim, dtype=torch.float64, device=self.device)
        self._ger_utt_gram = torch.zeros_like(self._ger_gram)
        self._ger_counts = torch.zeros(3, dtype=torch.float64, device=self.device)

    def _ger_update(self, batch):
        if self.ger_max_seconds <= 0 or batch is None:
            return

        world = self.trainer.world_size if self._trainer is not None else 1
        if float(self._ger_counts[2]) >= self.ger_max_seconds / max(1, world):
            return

        with torch.no_grad():
            hidden, _, padding_mask = self.model.extract_features(
                batch.inputs, batch.input_lengths, tgt_layer=self.ger_layer - 1
            )
        valid = ~padding_mask
        hidden = hidden.double()
        frames = hidden[valid]
        utterances = (hidden * valid.unsqueeze(-1)).sum(dim=1)
        self._ger_gram += frames.T @ frames
        self._ger_utt_gram += utterances.T @ utterances
        seconds = float(batch.input_lengths.sum()) / float(self.model.cfg.sample_rate)
        self._ger_counts += torch.tensor(
            [frames.size(0), utterances.size(0), seconds],
            dtype=torch.float64,
            device=self._ger_counts.device,
        )

    def on_validation_epoch_start(self):
        if self.ger_max_seconds > 0:
            self._ger_reset()

    def on_validation_epoch_end(self):
        if self.ger_max_seconds <= 0 or self.trainer.sanity_checking:
            return

        strategy = self.trainer.strategy
        gram = strategy.reduce(self._ger_gram, reduce_op="sum")
        utt_gram = strategy.reduce(self._ger_utt_gram, reduce_op="sum")
        counts = strategy.reduce(self._ger_counts, reduce_op="sum")
        if float(counts[0]) <= 0:
            return

        ger = effective_rank_from_gram(gram)
        rankme_t = effective_rank_from_gram(utt_gram)
        self.log("Metrics/val_ger", ger, sync_dist=True)
        self.log("Metrics/val_rankme_t", rankme_t, sync_dist=True)

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
    def __init__(self, args=None, sp_model=None, load_pretrained=True):
        super().__init__()
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
        self.random_init = bool(getattr(args, "random_init", False))
        pretrained_path = getattr(args, "pretrained_path", None)
        if self.random_init and pretrained_path:
            raise ValueError("random_init=True and pretrained_path are mutually exclusive")
        if self.random_init:
            rank_zero_info(
                f"[init] encoder: random init ({args.model_size}), no pretrained checkpoint loaded; "
                f"CNN feature extractor is trained"
            )
        elif load_pretrained and pretrained_path:
            self._load_pretrained(pretrained_path)
        self.save_hyperparameters(args)

        self.encoder.remove_pretraining_modules()
        if not self.random_init:
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

        trainable = [p for p in self.encoder.parameters() if p.requires_grad]
        trainable += list(self.ctc_out.parameters())
        self.optimizer = torch.optim.Adam(
            trainable,
            lr=float(args.lr),
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        # requires_grad is never toggled after this point: DDP only syncs parameters that
        # require grad when it wraps the model, so freezing is done with no_grad / grad=None.
        self.freeze_steps = int(getattr(args, "freeze_steps", 0) or 0)
        self.unfreeze_steps = int(getattr(args, "unfreeze_steps", 0) or 0)
        layers = self.encoder.encoder.layers
        layer_param_ids = {id(p) for p in layers.parameters()}
        self._unfreeze_groups = [
            [
                p
                for p in self.encoder.parameters()
                if p.requires_grad and id(p) not in layer_param_ids
            ]
        ] + [list(layer.parameters()) for layer in layers]

        self.train_wer = WordErrorRate()
        self.val_wer = WordErrorRate()
        self.test_wer = WordErrorRate()

    def _load_pretrained(self, path):
        report = load_encoder_state(self.encoder, path)
        pt_hparams = report["hparams"]
        pt_step = report["global_step"]
        pt_max_steps = pt_hparams.get("max_steps")
        progress = ""
        if pt_step is not None and pt_max_steps:
            progress = f" ({100.0 * pt_step / float(pt_max_steps):.1f}% of max_steps={pt_max_steps})"
        rank_zero_info(
            f"[init] encoder: loaded {report['loaded']}/{report['expected']} tensors from {path}; "
            f"pretrain model_size={pt_hparams.get('model_size')} "
            f"step={pt_step}{progress}; unexpected keys: {len(report['unexpected'])}"
        )
        if pt_step is not None and pt_max_steps and pt_step < 0.5 * float(pt_max_steps):
            rank_zero_info(
                f"[init] WARNING: pretrain checkpoint is at step {pt_step} of {pt_max_steps}; "
                f"encoder is likely under-trained"
            )
        self.args.pretrained_loaded_tensors = int(report["loaded"])
        self.args.pretrained_global_step = pt_step

    def _frozen_groups(self) -> int:
        total = len(self._unfreeze_groups)
        step = self.global_step
        if step < self.freeze_steps:
            return total
        if self.unfreeze_steps <= 0:
            return 0

        progress = (step - self.freeze_steps) / float(self.unfreeze_steps)
        if progress >= 1.0:
            return 0

        return total - 1 - int(progress * total)

    def on_before_optimizer_step(self, optimizer):
        n_frozen = self._frozen_groups()
        for group in self._unfreeze_groups[:n_frozen]:
            for parameter in group:
                parameter.grad = None

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
        frozen = step_type == "train" and self.global_step < self.freeze_steps
        with torch.no_grad() if frozen else contextlib.nullcontext():
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
        self.log(
            "Metrics/encoder_frozen_groups",
            float(self._frozen_groups()),
            on_step=True,
            on_epoch=False,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")
