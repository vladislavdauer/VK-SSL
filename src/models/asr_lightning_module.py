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
            "SPM vocab size must be equal to expected vocab size"
        )
        self.blank_idx = spm_vocab_size

        self.frontend = None
        self.encoder = conformer_v2_ctc_base()

        self.ctc_out = torch.nn.Linear(self.encoder.output_dim , spm_vocab_size + 1)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

        self.loss = torch.nn.CTCLoss(blank=self.blank_idx, reduction="sum")

        self.optimizer = torch.optim.AdamW(
            itertools.chain(*([self.encoder.parameters(), self.ctc_out.parameters()])),
            lr=3e-3,
            eps=1e-9,
            betas=(0.9, 0.98),
            weight_decay=1e-6
        )

    def _step(self, batch, _, step_type):
        if batch is None:
            return None

        features = batch.inputs
        output, src_lengths = self.encoder(features, batch.input_lengths)

        layer = self.ctc_out(output)
        probs = self.log_softmax(layer).transpose(0, 1)

        loss = self.loss(probs, batch.targets, src_lengths, batch.target_lengths)
        self.log(f"Losses/{step_type}_loss", loss, on_step=True, on_epoch=True)

        return loss

    def forward(self, batch):
        features = batch.inputs.to(self.device)
        encoder_out, src_lengths = self.encoder(features, batch.input_lengths.to(self.device))
        logits = self.ctc_out(encoder_out)
        log_probs = self.log_softmax(logits)

        predicted_ids = torch.argmax(log_probs, dim=-1)

        results = []
        for seq in predicted_ids:
            collapsed = []
            prev = self.blank_idx
            for idx in seq:
                if idx != prev and idx != self.blank_idx:
                    collapsed.append(idx.item())
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

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, "train")
        batch_size = batch.inputs.size(0)
        batch_sizes = self.all_gather(batch_size)

        if isinstance(batch_sizes, torch.Tensor) and batch_sizes.dim() > 0:
            loss *= batch_sizes.size(0) / batch_sizes.sum()

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

        self.frontend = None # audio_resnet()
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