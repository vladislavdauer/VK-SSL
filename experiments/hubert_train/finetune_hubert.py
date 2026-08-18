import pathlib
from argparse import ArgumentParser

import sentencepiece as spm

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from src.data.hubert_data_module import get_hubert_finetune_data_module
from src.models.hubert_lightning_module import HubertCTCModule


def run_train(args):
    seed_everything(1)
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
        save_top_k=3,
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
    trainer = Trainer(
        default_root_dir=args.exp_dir,
        logger=loggers,
        max_epochs=args.epochs,
        num_nodes=args.nodes,
        devices=(
            args.gpus if torch.cuda.is_available() else "auto"
            ),
        accelerator=(
            "gpu" if torch.cuda.is_available() else "auto"
            ),
        strategy=(
            DDPStrategy(find_unused_parameters=False) if torch.cuda.is_available() else "auto"
            ),
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=0,
        gradient_clip_val=0.0,
        limit_train_batches=(50 if args.sanity_check else None),
        limit_val_batches=(10 if args.sanity_check else None),
        accumulate_grad_batches=args.accumulate_grad_batches,
        enable_progress_bar=True,
    )

    sp_model = spm.SentencePieceProcessor(model_file=str(args.sp_model_path))
    model = HubertCTCModule(args, sp_model)

    data_module = get_hubert_finetune_data_module(
        str(args.librispeech_path),
        str(args.sp_model_path),
        sanity_check=bool(args.sanity_check),
        durations_cache_dir=str(args.durations_cache_dir)
        if args.durations_cache_dir
        else None,
        num_workers=args.num_workers,
        max_batch_duration=float(args.max_batch_duration),
        )
    trainer.fit(model, data_module, ckpt_path=args.checkpoint_path)


def cli_main():
    parser = ArgumentParser()
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
        type=pathlib.Path,
        help="Path to SentencePiece model.",
        required=True,
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
        default=50.0,
        type=float,
        help="Label rate used at pre-train, needed to rebuild the encoder. (Default: 50)",
    )
    parser.add_argument(
        "--mask-alpha",
        default=1.0,
        type=float,
        help="Unused at fine-tune, kept to rebuild the encoder config. (Default: 1.0)",
    )
    parser.add_argument(
        "--lr",
        default=5.0,
        type=float,
        help="Peak learning rate for Noam/AdamW. (Default: 5.0)",
    )
    parser.add_argument(
        "--freeze-steps",
        default=0,
        type=int,
        help="Steps to train only the CTC head before unfreezing the transformer. (Default: 0)",
    )
    parser.add_argument(
        "--warmup-steps",
        default=10000,
        type=int,
        help="Noam warmup steps. (Default: 10000)",
    )
    parser.add_argument(
        "--max-batch-duration",
        default=200.0,
        type=float,
        help="Max seconds of audio per GPU batch. (Default: 200)",
    )
    parser.add_argument(
        "--nodes",
        default=1,
        type=int,
        help="Number of nodes to use for training. (Default: 1)",
    )
    parser.add_argument(
        "--gpus",
        default=2,
        type=int,
        help="Number of GPUs per node to use for training. (Default: 2)",
    )
    parser.add_argument(
        "--epochs",
        default=150,
        type=int,
        help="Number of epochs to train for. (Default: 150)",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        default=1,
        type=int,
        help="Gradient accumulation steps. (Default: 1)",
    )
    parser.add_argument(
        "--num-workers",
        default=4,
        type=int,
        help="DataLoader workers per process. Use 0-2 on slow NFS. (Default: 4)",
    )
    parser.add_argument(
        "--sanity_check",
        action="store_true",
        help="Run sanity check with small subset of data.",
    )
    args = parser.parse_args()
    run_train(args)


if __name__ == "__main__":
    cli_main()
