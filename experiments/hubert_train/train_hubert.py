import pathlib
import argparse
from argparse import ArgumentParser

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from src.data.hubert_data_module import get_hubert_pretrain_data_module
from src.models.hubert_lightning_module import HubertPretrainModule

PRETRAIN_REF_GPUS = 32


def _scaled_accum(reference_gpus, gpus, override=None):
    if override is not None:
        return int(override)
    return max(1, reference_gpus // max(1, int(gpus)))


def run_train(args):
    seed_everything(1)
    accumulate_grad_batches = _scaled_accum(
        PRETRAIN_REF_GPUS,
        args.gpus,
        args.accumulate_grad_batches,
    )
    checkpoint_dir = args.exp_dir / "checkpoints"
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/val_loss",
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
        gradient_clip_val=float(args.gradient_clip_val),
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

    model = HubertPretrainModule(args)
    dummy = bool(args.sanity_check and not args.label_paths)
    data_module = get_hubert_pretrain_data_module(
        str(args.librispeech_path),
        label_paths=[str(p) for p in args.label_paths] if args.label_paths else None,
        dummy_labels=dummy,
        num_classes=[int(v) for v in str(args.num_classes).split(",") if str(v).strip()],
        label_rate=float(args.label_rate),
        sanity_check=bool(args.sanity_check),
        durations_cache_dir=str(args.durations_cache_dir)
        if args.durations_cache_dir
        else None,
        num_workers=args.num_workers,
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
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--exp-dir",
        default=pathlib.Path("./exp_hubert"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp_hubert')",
    )
    parser.add_argument(
        "--librispeech-path",
        type=pathlib.Path,
        help="Path to LibriSpeech datasets.",
        required=True,
    )
    parser.add_argument(
        "--label-paths",
        nargs="*",
        default=None,
        type=pathlib.Path,
        help="One or more .km label files. Optional with --sanity_check.",
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
        help="HuBERT size. (Default: base)",
    )
    parser.add_argument(
        "--num-classes",
        default="100",
        help="Comma-separated codebook sizes, e.g. '100' or '100,500'. (Default: 100)",
    )
    parser.add_argument(
        "--label-rate",
        default=100.0,
        type=float,
        help="Teacher label rate in Hz. Use 100 for MFCC, 50 for CNN/transformer features. (Default: 100)",
    )
    parser.add_argument(
        "--mask-alpha",
        default=1.0,
        type=float,
        help="Weight of masked-frame loss. 1.0 is masked-only. (Default: 1.0)",
    )
    parser.add_argument(
        "--mask-prob",
        default=0.80,
        type=float,
        help="Masked frame fraction. (Default: 0.80)",
    )
    parser.add_argument(
        "--fairseq-mask",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Fairseq-style mask counting. (Default: True)",
    )
    parser.add_argument(
        "--feature-grad-mult",
        default=0.1,
        type=float,
        help="CNN gradient multiplier. (Default: 0.1)",
    )
    parser.add_argument(
        "--lr",
        default=5e-4,
        type=float,
        help="Peak learning rate. (Default: 5e-4)",
    )
    parser.add_argument(
        "--weight-decay",
        default=0.01,
        type=float,
        help="Adam weight decay. (Default: 0.01)",
    )
    parser.add_argument(
        "--warmup-ratio",
        default=0.08,
        type=float,
        help="LR warmup fraction. (Default: 0.08)",
    )
    parser.add_argument(
        "--gradient-clip-val",
        default=10.0,
        type=float,
        help="Gradient clip norm. (Default: 10.0)",
    )
    parser.add_argument(
        "--max-batch-duration",
        default=87.5,
        type=float,
        help="Max seconds of audio per GPU batch. (Default: 87.5)",
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
        help="Gradient accumulation. Default: 32 // gpus.",
    )
    parser.add_argument(
        "--epochs",
        default=1,
        type=int,
        help="Number of epochs if --max-steps is not set. (Default: 1)",
    )
    parser.add_argument(
        "--max-steps",
        default=250000,
        type=int,
        help="Train for this many optimizer steps. iter1: 250000, iter2: 400000. (Default: 250000)",
    )
    parser.add_argument(
        "--num-workers",
        default=6,
        type=int,
        help="DataLoader workers. (Default: 6)",
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
        default=["train-clean-100", "train-clean-360", "train-other-500"],
        help="LibriSpeech train splits for pretrain. Default: full ~960h (~1k).",
    )
    args = parser.parse_args()
    run_train(args)


if __name__ == "__main__":
    cli_main()
