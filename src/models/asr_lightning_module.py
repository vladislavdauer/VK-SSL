import itertools
import math

from collections import namedtuple
from typing import List, Tuple

import sentencepiece as spm

import torch
import torchaudio
from pytorch_lightning import LightningModule

from src.models.build_model import conformer_rnnt_base, conformer_v2_ctc_base
from src.opt.schedulers import WarmupCosineScheduler
from src.models.rnnt_decoder import Hypothesis, RNNTBeamSearch
from torchmetrics.text import WordErrorRate

_expected_spm_vocab_size = 1023

Batch = namedtuple("Batch", ["inputs", "input_lengths", "targets", "target_lengths"])


class CTCTModule(LightningModule):
    def __init__(self, args=None, sp_model=None, pretrained_model_path=None):
        super().__init__()
        self.warmup_lr_scheduler = None
        self.save_hyperparameters(args)
        self.args = args
        self.sp_model = sp_model
        spm_vocab_size = self.sp_model.get_piece_size()

        assert spm_vocab_size == _expected_spm_vocab_size, (
            f"SPM vocab size ({spm_vocab_size}) must be equal to expected vocab size ({_expected_spm_vocab_size})"
        )
        self.blank_idx = spm_vocab_size

        self.frontend = None
        self.encoder = conformer_v2_ctc_base()

        self.ctc_out = torch.nn.Linear(self.encoder.output_dim, spm_vocab_size + 1)
        torch.nn.init.xavier_uniform_(self.ctc_out.weight)
        torch.nn.init.constant_(self.ctc_out.bias, 0.0)

        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

        self.loss = torch.nn.CTCLoss(blank=self.blank_idx, reduction="none", zero_infinity=True, )

        self.optimizer = torch.optim.AdamW(
            itertools.chain(*([self.encoder.parameters(), self.ctc_out.parameters()])),
            lr=3e-3,
            eps=1e-9,
            betas=(0.9, 0.98),
            weight_decay=1e-6
        )

        self.train_wer = WordErrorRate()
        self.val_wer = WordErrorRate()
        self.test_wer = WordErrorRate()

    def _step(self, batch, _, step_type):
        if batch is None:
            return None

        features = batch.inputs
        output, src_lengths = self.encoder(features, batch.input_lengths)

        layer = self.ctc_out(output)
        probs = self.log_softmax(layer).transpose(0, 1)

        loss = self.loss(
            probs,
            batch.targets,
            src_lengths,
            batch.target_lengths,
        ).mean()

        self.log(f"Losses/{step_type}_loss", loss, on_epoch=True)

        if step_type in ("train", "val", "test"):
            with torch.no_grad():
                pred_ids = probs.transpose(0, 1).argmax(dim=-1)

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

                    pred_texts.append(self.sp_model.decode(collapsed))

                    target_len = int(batch.target_lengths[i].item())
                    target_ids = (
                        batch.targets[i, :target_len]
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    target_texts.append(self.sp_model.decode(target_ids))

                if step_type == "train":
                    self.train_wer.update(pred_texts, target_texts)
                    self.log(
                        "Metrics/train_wer",
                        self.train_wer,
                        on_step=False,
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
        encoder_out, src_lengths = self.encoder(features, batch.input_lengths.to(self.device))
        logits = self.ctc_out(encoder_out)
        log_probs = self.log_softmax(logits)

        predicted_ids = torch.argmax(log_probs, dim=-1)

        results = []
        for seq, length in zip(predicted_ids, src_lengths):
            seq = seq[:length]

            collapsed = []
            prev = self.blank_idx

            for idx in seq:
                idx = idx.item()
                if idx != prev and idx != self.blank_idx:
                    collapsed.append(idx)
                prev = idx

            results.append(self.sp_model.decode(collapsed))

        return results[0] if len(results) == 1 else results

    def configure_optimizers(self):
        if self.trainer is not None:
            total_steps = self.trainer.estimated_stepping_batches
        else:
            total_steps = 10000

        self.warmup_lr_scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=10,
            total_epochs=self.args.epochs,
            steps_per_epoch=total_steps // self.args.epochs,
        )

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.warmup_lr_scheduler,
                "interval": "step",
            }
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
        loss = self._step(batch, batch_idx, "train")

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
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")


def post_process_hypos(
        hypos: List[Hypothesis], sp_model: spm.SentencePieceProcessor
) -> List[Tuple[str, float, List[int], List[int]]]:
    tokens_idx = 0
    score_idx = 3
    post_process_remove_list = [
        sp_model.unk_id(),
        sp_model.eos_id(),
        sp_model.pad_id(),
    ]
    filtered_hypo_tokens = [
        [token_index for token_index in h[tokens_idx][1:] if token_index not in post_process_remove_list] for h in hypos
    ]
    hypos_str = [sp_model.decode(s) for s in filtered_hypo_tokens]
    hypos_ids = [h[tokens_idx][1:] for h in hypos]
    hypos_score = [[math.exp(h[score_idx])] for h in hypos]

    nbest_batch = list(zip(hypos_str, hypos_score, hypos_ids))

    return nbest_batch


class ConformerRNNTModule(LightningModule):

    def __init__(self, args=None, sp_model=None, pretrained_model_path=None):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.sp_model = sp_model
        spm_vocab_size = self.sp_model.get_piece_size()
        assert spm_vocab_size == _expected_spm_vocab_size, (
            "The model returned by conformer_rnnt_base expects a SentencePiece model of "
            f"vocabulary size {_expected_spm_vocab_size}, but the given SentencePiece model has a vocabulary size "
            f"of {spm_vocab_size}. Please provide a correctly configured SentencePiece model."
        )
        self.blank_idx = spm_vocab_size

        self.frontend = None  # audio_resnet()
        self.model = conformer_rnnt_base()

        # TODO: FIXME
        # -- initialise
        # if args.pretrained_model_path:
        #     ckpt = torch.load(args.pretrained_model_path, map_location=lambda storage, loc: storage)
        #     tmp_ckpt = {
        #         k.replace("encoder.frontend.", ""): v for k, v in ckpt.items() if k.startswith("encoder.frontend.")
        #     }
        #     self.frontend.load_state_dict(tmp_ckpt)

        self.loss = torchaudio.transforms.RNNTLoss(reduction="sum")

        self.optimizer = torch.optim.AdamW(
            # itertools.chain(*([self.frontend.parameters(), self.model.parameters()])),
            itertools.chain(*([self.model.parameters()])),
            lr=8e-4,
            weight_decay=0.06,
            betas=(0.9, 0.98),
        )

    def _step(self, batch, _, step_type):
        if batch is None:
            return None

        prepended_targets = batch.targets.new_empty([batch.targets.size(0), batch.targets.size(1) + 1])
        prepended_targets[:, 1:] = batch.targets
        prepended_targets[:, 0] = self.blank_idx
        prepended_target_lengths = batch.target_lengths + 1
        # features = self.frontend(batch.inputs)
        features = batch.inputs
        output, src_lengths, _, _ = self.model(
            features, batch.input_lengths, prepended_targets, prepended_target_lengths
        )
        loss = self.loss(output, batch.targets, src_lengths, batch.target_lengths)
        self.log(f"Losses/{step_type}_loss", loss, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        self.warmup_lr_scheduler = WarmupCosineScheduler(
            self.optimizer,
            10,
            self.args.epochs,
            len(self.trainer.datamodule.train_dataloader()) / self.trainer.num_devices / self.trainer.num_nodes,
        )
        self.lr_scheduler_interval = "step"
        return (
            [self.optimizer],
            [{"scheduler": self.warmup_lr_scheduler, "interval": self.lr_scheduler_interval}],
        )

    def forward(self, batch):
        decoder = RNNTBeamSearch(self.model, self.blank_idx)
        # x = self.frontend(batch.inputs.to(self.device))
        x = batch.inputs.to(self.device)
        hypotheses = decoder(x, batch.input_lengths.to(self.device), beam_width=20)
        return post_process_hypos(hypotheses, self.sp_model)[0][0]

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, "train")
        batch_size = batch.inputs.size(0)
        batch_sizes = self.all_gather(batch_size)
        if isinstance(batch_sizes, torch.Tensor) and batch_sizes.dim() > 0:
            loss *= batch_sizes.size(0) / batch_sizes.sum()  # world size / batch size
        self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))
        return loss

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")
