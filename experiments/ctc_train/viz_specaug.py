import argparse
import pathlib

import matplotlib.pyplot as plt
import torch
import torchaudio

from src.data.data_transforms import TrainTransform, ValTransform

def _to_btf(features: torch.Tensor, length: int) -> torch.Tensor:
    return features[0, :length].transpose(0, 1).detach().cpu()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--librispeech-path",
        type=pathlib.Path,
        required=True
    )
    parser.add_argument(
        "--global-stats-path",
        type=pathlib.Path,
        required=True
    )
    parser.add_argument(
        "--sp-model-path",
        type=pathlib.Path,
        required=True
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("./specaug_viz")
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4
    )
    parser.add_argument(
        "--aug-draws",
        type=int,
        default=3
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="dev-clean"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    dataset = torchaudio.datasets.LIBRISPEECH(
        str(args.librispeech_path), url=args.subset, download=False
    )
    val_tf = ValTransform(str(args.global_stats_path), str(args.sp_model_path))
    train_tf = TrainTransform(str(args.global_stats_path), str(args.sp_model_path))

    for i in range(args.num_samples):
        sample = dataset[i]
        transcript = sample[2]

        clean_batch = val_tf([sample])
        clean = _to_btf(clean_batch.inputs, int(clean_batch.input_lengths[0]))
        t_frames = clean.size(1)

        n_cols = 1 + args.aug_draws
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), sharey=True)
        if n_cols == 1:
            axes = [axes]

        axes[0].imshow(clean.numpy(), aspect="auto", origin="lower", cmap="magma")
        axes[0].set_title("clean (val pipeline)")
        axes[0].set_xlabel("time frames")
        axes[0].set_ylabel("mel bins")

        for j in range(args.aug_draws):
            aug_batch = train_tf([sample])
            aug = _to_btf(aug_batch.inputs, int(aug_batch.input_lengths[0]))
            diff = (aug - clean).abs()
            changed = (diff > 1e-3).float().mean().item() * 100

            axes[j + 1].imshow(aug.numpy(), aspect="auto", origin="lower", cmap="magma")
            axes[j + 1].set_title(f"SpecAug draw {j + 1}\n~{changed:.0f}% bins changed")
            axes[j + 1].set_xlabel("time frames")

        fig.suptitle(f"[{i}] T={t_frames} | {transcript[:80]}", fontsize=10)
        fig.tight_layout()
        out_path = args.out_dir / f"sample_{i:03d}.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"saved {out_path} (T={t_frames})")

    print(
        "Current train SpecAug (~NeMo Medium): "
        "2x FrequencyMasking(27) + 5x TimeMasking(10000, p=0.05)."
    )

if __name__ == "__main__":
    main()
