import pathlib
from argparse import ArgumentParser

import sentencepiece as spm

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from src.models.asr_lightning_module import ConformerRNNTModule
from src.data.librispeech_data_module import get_data_module

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
    trainer = Trainer(
        default_root_dir=args.exp_dir,
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
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=10.0,
        limit_train_batches=(50 if args.sanity_check else None), 
        limit_val_batches=(10 if args.sanity_check else None),
    )

    sp_model = spm.SentencePieceProcessor(model_file=str(args.sp_model_path))
    model = ConformerRNNTModule(args, sp_model)
    data_module = get_data_module(
        str(args.librispeech_path), 
        str(args.global_stats_path), 
        str(args.sp_model_path),
        sanity_check=bool(args.sanity_check),
        )
    trainer.fit(model, data_module, ckpt_path=args.checkpoint_path)

def cli_main():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        type=pathlib.Path,
        help="Path to checkpoint to use for evaluation.",
    )
    parser.add_argument(
        "--exp-dir",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--global-stats-path",
        default=pathlib.Path("global_stats.json"),
        type=pathlib.Path,
        help="Path to JSON file containing feature means and stddevs.",
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
        "--nodes",
        default=1,
        type=int,
        help="Number of nodes to use for training. (Default: 4)",
    )
    parser.add_argument(
        "--gpus",
        default=2,
        type=int,
        help="Number of GPUs per node to use for training. (Default: 8)",
    )
    parser.add_argument(
        "--epochs",
        default=120,
        type=int,
        help="Number of epochs to train for. (Default: 120)",
    )
    parser.add_argument(
        "--sanity_check",
        action="store_true",
    )
    args = parser.parse_args()
    run_train(args)


if __name__ == "__main__":
    cli_main()