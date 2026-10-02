import argparse
import pathlib
from argparse import ArgumentParser

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from src.data.hubert_data_module import get_hubert_finetune_data_module
from src.models.hubert_lightning_module import HubertCTCModule

FT_PRESETS = {
    "100h": dict(
        train_subsets=["train-clean-100"],
        lr=3e-5,
        max_steps=80000,
        warmup_steps=8000,
        hold_steps=32000,
        decay_steps=40000,
        ft_mask_prob=0.065,
        batch_seconds=1600.0,
    ),
    "10h": dict(
        train_subsets=["ll-10h"],
        lr=2e-5,
        max_steps=25000,
        warmup_steps=8000,
        hold_steps=0,
        decay_steps=72000,
        ft_mask_prob=0.075,
        batch_seconds=200.0,
    ),
}
MAX_GPU_BATCH_SECONDS = 200.0

def apply_preset(args):
    preset = FT_PRESETS[args.preset]
    for key, value in preset.items():
        if key != "batch_seconds" and getattr(args, key) is None:
            setattr(args, key, value)

    world = max(1, int(args.gpus) * int(args.nodes))
    if args.max_batch_duration is None:
        args.max_batch_duration = min(MAX_GPU_BATCH_SECONDS, preset["batch_seconds"] / world)
    if args.accumulate_grad_batches is None:
        per_step = world * float(args.max_batch_duration)
        args.accumulate_grad_batches = max(1, round(preset["batch_seconds"] / per_step))

    return args

def run_train(args):
    seed_everything(1)
    args = apply_preset(args)
    accum = int(args.accumulate_grad_batches)
    print(
        f"preset={args.preset} subsets={args.train_subsets} lr={args.lr} steps={args.max_steps} "
        f"batch={args.gpus * args.nodes}x{args.max_batch_duration:g}s x accum {accum}",
        flush=True,
    )

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
        precision="32-true",
        gradient_clip_val=0.0,
        limit_train_batches=(50 if args.sanity_check else None),
        limit_val_batches=(10 if args.sanity_check else None),
        accumulate_grad_batches=accum,
        enable_progress_bar=True,
    )
    if args.max_steps is not None:
        trainer_kwargs["max_steps"] = int(args.max_steps)
        trainer_kwargs["max_epochs"] = -1
    else:
        trainer_kwargs["max_epochs"] = args.epochs

    trainer = Trainer(**trainer_kwargs)

    sp_model = None
    if args.label_type == "spm":
        import sentencepiece as spm

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
        max_batch_duration=float(args.max_batch_duration),
        train_subsets=args.train_subsets,
        val_subsets=args.val_subsets,
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
        "--preset",
        default="100h",
        choices=list(FT_PRESETS),
        help=(
            "Labelled-data recipe: '100h' = train-clean-100, '10h' = Libri-Light 10h "
            "(<librispeech-path>/librispeech_finetuning). Sets data, LR schedule, steps, "
            "mask and batch; any flag passed explicitly wins. (Default: 100h)"
        ),
    )
    parser.add_argument(
        "--train-subsets",
        nargs="+",
        default=None,
        help="Override training splits, e.g. train-clean-100 or ll-10h / ll-1h / ll-10min.",
    )
    parser.add_argument(
        "--val-subsets",
        nargs="+",
        default=["dev-clean", "dev-other"],
        help="Validation splits. (Default: dev-clean dev-other)",
    )
    parser.add_argument(
        "--label-type",
        default="char",
        choices=["char", "spm"],
        help="CTC targets. fairseq fine-tunes on letters. (Default: char)",
    )
    parser.add_argument(
        "--sp-model-path",
        default=None,
        type=pathlib.Path,
        help="SentencePiece model, required only for --label-type spm.",
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
        help="Codebook sizes used at pre-train, needed to rebuild the config. (Default: 100)",
    )
    parser.add_argument(
        "--label-rate",
        default=100.0,
        type=float,
        help="Pre-train label rate, needed to rebuild the config. (Default: 100)",
    )
    parser.add_argument(
        "--mask-alpha",
        default=1.0,
        type=float,
        help="Unused at fine-tune, kept to rebuild the encoder config. (Default: 1.0)",
    )
    parser.add_argument(
        "--lr",
        default=None,
        type=float,
        help="Peak LR. (Preset: 100h 3e-5, 10h 2e-5)",
    )
    parser.add_argument(
        "--max-steps",
        default=None,
        type=int,
        help="Optimizer steps. (Preset: 100h 80000, 10h 25000)",
    )
    parser.add_argument(
        "--warmup-steps",
        default=None,
        type=int,
        help="Tri-stage warmup. (Preset: 8000 for both)",
    )
    parser.add_argument(
        "--hold-steps",
        default=None,
        type=int,
        help="Tri-stage hold at peak LR. (Preset: 100h 32000, 10h 0)",
    )
    parser.add_argument(
        "--decay-steps",
        default=None,
        type=int,
        help="Tri-stage exponential decay. (Preset: 100h 40000, 10h 72000)",
    )
    parser.add_argument(
        "--final-lr-scale",
        default=0.05,
        type=float,
        help="Final LR as a fraction of peak. (Default: 0.05)",
    )
    parser.add_argument(
        "--freeze-steps",
        default=10000,
        type=int,
        help="Train only the CTC head for this many steps. (Default: 10000)",
    )
    parser.add_argument(
        "--apply-ft-mask",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Span + channel masking during fine-tune. (Default: True)",
    )
    parser.add_argument(
        "--ft-mask-prob",
        default=None,
        type=float,
        help=(
            "Fraction of frames used as mask starts, = fairseq mask_prob / span 10. "
            "(Preset: 100h 0.065, 10h 0.075)"
        ),
    )
    parser.add_argument(
        "--ft-mask-channel-prob",
        default=0.5,
        type=float,
        help="Channel mask probability in fairseq units. (Default: 0.5)",
    )
    parser.add_argument(
        "--ft-mask-channel-length",
        default=64,
        type=int,
        help="Channel mask span. (Default: 64)",
    )
    parser.add_argument(
        "--dropout",
        default=0.0,
        type=float,
        help="Transformer dropout at fine-tune. (Default: 0.0)",
    )
    parser.add_argument(
        "--attention-dropout",
        default=0.0,
        type=float,
        help="Attention dropout at fine-tune. (Default: 0.0)",
    )
    parser.add_argument(
        "--activation-dropout",
        default=0.1,
        type=float,
        help="FFN activation dropout at fine-tune. (Default: 0.1)",
    )
    parser.add_argument(
        "--final-dropout",
        default=0.0,
        type=float,
        help="Dropout before the CTC projection. (Default: 0.0)",
    )
    parser.add_argument(
        "--layerdrop",
        default=0.1,
        type=float,
        help="Layerdrop at fine-tune. (Default: 0.1)",
    )
    parser.add_argument(
        "--max-batch-duration",
        default=None,
        type=float,
        help=(
            "Max seconds of audio per GPU batch. Default: preset batch / GPUs, capped at "
            "200 s (= fairseq max_tokens 3200000)."
        ),
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
        help="Gradient accumulation. Default: whatever reaches the preset batch (100h 1600 s, 10h 200 s).",
    )
    parser.add_argument(
        "--epochs",
        default=150,
        type=int,
        help="Used only when --max-steps is not set. (Default: 150)",
    )
    parser.add_argument(
        "--num-workers",
        default=4,
        type=int,
        help="DataLoader workers per process. (Default: 4)",
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
