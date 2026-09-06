import pathlib
import warnings
import argparse
from argparse import ArgumentParser

import sentencepiece as spm

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from src.data.hubert_data_module import get_hubert_finetune_data_module
from src.models.hubert_lightning_module import HubertCTCModule

FT_REF_GPUS = 8

warnings.filterwarnings(
    "ignore",
    message="Detected call of `lr_scheduler.step()` before `optimizer.step()`",
)


def _scaled_accum(reference_gpus, gpus, override=None):
    if override is not None:
        return int(override)
    return max(1, reference_gpus // max(1, int(gpus)))


def run_train(args):
    seed_everything(1)
    accumulate_grad_batches = _scaled_accum(
        FT_REF_GPUS,
        args.gpus,
        args.accumulate_grad_batches,
    )
    decay_steps = int(args.decay_steps)
    if decay_steps <= 0:
        decay_steps = 72000

    checkpoint_dir = args.exp_dir / "checkpoints"
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Metrics/val_wer",
        mode="min",
        save_top_k=5,
        save_weights_only=False,
        verbose=True,
    )
    train_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/train_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=False,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [
        checkpoint,
        train_checkpoint,
        lr_monitor,
    ]
    tb_logger = TensorBoardLogger(save_dir=args.exp_dir, name="lightning_logs", version=None)
    loggers = [
        tb_logger,
        CSVLogger(save_dir=args.exp_dir, name="lightning_logs", version=tb_logger.version),
    ]
    trainer_kwargs = dict(
        default_root_dir=args.exp_dir,
        logger=loggers,
        num_nodes=args.nodes,
        devices=(
            args.gpus if torch.cuda.is_available() else "auto"
            ),
        accelerator=(
            "gpu" if torch.cuda.is_available() else "auto"
            ),
        strategy=(
            DDPStrategy(find_unused_parameters=True) if torch.cuda.is_available() else "auto"
            ),
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=0,
        precision=args.precision,
        gradient_clip_val=0.0,
        limit_train_batches=(50 if args.sanity_check else None),
        limit_val_batches=(10 if args.sanity_check else None),
        accumulate_grad_batches=accumulate_grad_batches,
        enable_progress_bar=True,
    )
    if args.max_steps is not None:
        trainer_kwargs["max_steps"] = int(args.max_steps)
        trainer_kwargs["max_epochs"] = -1
    else:
        trainer_kwargs["max_epochs"] = args.epochs

    trainer = Trainer(**trainer_kwargs)

    args.decay_steps = decay_steps
    sp_model = None
    if args.label_type == "spm":
        sp_model = spm.SentencePieceProcessor(model_file=str(args.sp_model_path))
    model = HubertCTCModule(args, sp_model)

    data_module = get_hubert_finetune_data_module(
        str(args.librispeech_path),
        sp_model_path=str(args.sp_model_path) if args.sp_model_path else None,
        label_type=args.label_type,
        sanity_check=bool(args.sanity_check),
        durations_cache_dir=str(args.durations_cache_dir)
        if args.durations_cache_dir
        else None,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        max_batch_duration=float(args.max_batch_duration),
        train_subsets=args.train_subsets,
        )
    trainer.fit(model, data_module, ckpt_path=args.checkpoint_path)


def cli_main():
    parser = ArgumentParser(
            )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        type=pathlib.Path,
        help="Path to checkpoint to resume fine-tuning from.",
    )
    parser.add_argument(
        "--pretrained-path",
        default=None,
        type=pathlib.Path,
        help="HuBERT pre-train checkpoint to initialize the encoder from.",
    )
    parser.add_argument(
        "--exp-dir",
        default=pathlib.Path("./exp_hubert_ft"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp_hubert_ft')",
    )
    parser.add_argument(
        "--librispeech-path",
        type=pathlib.Path,
        help="Path to LibriSpeech datasets.",
        required=True,
    )
    parser.add_argument(
        "--sp-model-path",
        default=None,
        type=pathlib.Path,
        help="SentencePiece model (required for --label-type spm).",
    )
    parser.add_argument(
        "--label-type",
        default="char",
        choices=["char", "spm"],
        help="CTC targets: char or spm. (Default: char)",
    )
    parser.add_argument(
        "--durations-cache-dir",
        default=None,
        type=pathlib.Path,
        help="JSON cache for audio durations (default: <librispeech>/.duration_cache).",
    )
    parser.add_argument(
        "--model-size",
        default="base",
        choices=["tiny", "base", "large", "xlarge"],
        help="HuBERT size. Must match the pre-train checkpoint. (Default: base)",
    )
    parser.add_argument(
        "--num-classes",
        default="100",
        help="Comma-separated codebook sizes used at pre-train. (Default: 100)",
    )
    parser.add_argument(
        "--label-rate",
        default=100.0,
        type=float,
        help="Pretrain label rate. (Default: 100)",
    )
    parser.add_argument(
        "--mask-alpha",
        default=1.0,
        type=float,
        help="Unused at fine-tune, kept to rebuild the encoder config. (Default: 1.0)",
    )
    parser.add_argument(
        "--lr",
        default=2e-5,
        type=float,
        help="Peak LR. (Default: 2e-5)",
    )
    parser.add_argument(
        "--freeze-steps",
        default=10000,
        type=int,
        help="Train only CTC head for this many optimizer steps. (Default: 10000)",
    )
    parser.add_argument(
        "--warmup-steps",
        default=8000,
        type=int,
        help="Tri-stage warmup steps. (Default: 8000)",
    )
    parser.add_argument(
        "--hold-steps",
        default=0,
        type=int,
        help="Tri-stage hold steps at peak LR. (Default: 0)",
    )
    parser.add_argument(
        "--decay-steps",
        default=72000,
        type=int,
        help="Tri-stage decay steps. (Default: 72000)",
    )
    parser.add_argument(
        "--final-lr-scale",
        default=0.05,
        type=float,
        help="Final LR as a fraction of peak LR. (Default: 0.05)",
    )
    parser.add_argument(
        "--max-steps",
        default=25000,
        type=int,
        help="Optimizer steps. (Default: 25000)",
    )
    parser.add_argument(
        "--max-batch-duration",
        default=200.0,
        type=float,
        help="Max seconds of audio per GPU batch. (Default: 200)",
    )
    parser.add_argument(
        "--batch-size",
        default=None,
        type=int,
        help="Optional utterance batch size. If set, overrides max-batch-duration.",
    )
    parser.add_argument(
        "--apply-ft-mask",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Span masking during fine-tune. (Default: True)",
    )
    parser.add_argument(
        "--ft-mask-prob",
        default=0.75,
        type=float,
        help="Fine-tune time-mask probability. (Default: 0.75)",
    )
    parser.add_argument(
        "--ft-mask-channel-prob",
        default=0.5,
        type=float,
        help="Fine-tune channel-mask probability. (Default: 0.5)",
    )
    parser.add_argument(
        "--ft-mask-channel-length",
        default=64,
        type=int,
        help="Fine-tune channel-mask span. (Default: 64)",
    )
    parser.add_argument(
        "--layerdrop",
        default=0.1,
        type=float,
        help="Layerdrop. (Default: 0.1)",
    )
    parser.add_argument(
        "--nodes",
        default=1,
        type=int,
        help="Number of nodes to use for training. (Default: 1)",
    )
    parser.add_argument(
        "--gpus",
        default=4,
        type=int,
        help="GPUs per node. (Default: 4)",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        default=None,
        type=int,
        help="Gradient accumulation. Default: 8 // gpus.",
    )
    parser.add_argument(
        "--epochs",
        default=150,
        type=int,
        help="Used only when --max-steps is not set. (Default: 150)",
    )
    parser.add_argument(
        "--num-workers",
        default=0,
        type=int,
        help="DataLoader workers. (Default: 0)",
    )
    parser.add_argument(
        "--sanity_check",
        action="store_true",
        help="Run sanity check with small subset of data.",
    )
    parser.add_argument(
        "--precision",
        default="16-mixed",
        choices=["32-true", "16-mixed", "bf16-mixed"],
        help="Training precision. (Default: 16-mixed)",
    )
    parser.add_argument(
        "--train-subsets",
        nargs="+",
        default=["train-clean-100"],
        help="Train subsets. (Default: train-clean-100)",
    )
    args = parser.parse_args()
    run_train(args)


if __name__ == "__main__":
    cli_main()
