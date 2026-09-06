import json
import pathlib
from argparse import ArgumentParser

import joblib
import numpy as np
from tqdm import tqdm

from src.models.hubert.kmeans import fit_kmeans


def run_learn(args):
    with tqdm(total=1, desc="load features.npy", unit="file") as bar:
        features = np.load(args.features_path, mmap_mode="r")
        bar.update(1)

    file_gib = features.shape[0] * features.shape[1] * np.dtype(features.dtype).itemsize / 2 ** 30
    tqdm.write(
        f"mapped {features.shape[0]} frames x {features.shape[1]} "
        f"({file_gib:.2f} GiB on disk, mmap)"
    )

    with open(args.index_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    model = fit_kmeans(
        features,
        n_clusters=int(args.n_clusters),
        percent=float(args.percent),
        batch_size=int(args.batch_size),
        n_init=int(args.n_init),
        max_iter=int(args.max_iter),
        seed=int(args.seed),
        max_frames=None if int(args.max_frames) < 0 else int(args.max_frames),
        show_progress=True,
    )

    args.km_path.parent.mkdir(parents=True, exist_ok=True)
    with tqdm(total=1, desc="save km model", unit="file") as bar:
        joblib.dump(model, args.km_path)
        bar.update(1)

    fit_stats = getattr(model, "_hubert_fit_stats", {})
    info = {
        "n_clusters": int(args.n_clusters),
        "percent": float(args.percent),
        "max_frames": int(args.max_frames),
        "batch_size": int(args.batch_size),
        "n_init": int(args.n_init),
        "max_iter": int(args.max_iter),
        "seed": int(args.seed),
        "feature_type": meta.get("feature_type"),
        "layer": meta.get("layer"),
        "label_rate": meta.get("label_rate"),
        "n_frames": int(features.shape[0]),
        "dim": int(features.shape[1]) if features.ndim == 2 else 0,
        "fit": fit_stats,
    }
    with open(args.km_path.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)

    tqdm.write(f"saved {args.km_path}")
    if fit_stats:
        tqdm.write(
            f"quality: used={fit_stats.get('used_clusters')}/{args.n_clusters} "
            f"entropy={fit_stats.get('entropy'):.3f} "
            f"inertia/sample={fit_stats.get('inertia_per_sample'):.5f}"
        )


def cli_main():
    parser = ArgumentParser()
    parser.add_argument(
        "--features-path",
        type=pathlib.Path,
        help="Path to features.npy from extract_features.py.",
        required=True,
    )
    parser.add_argument(
        "--index-path",
        type=pathlib.Path,
        help="Path to index.json from extract_features.py.",
        required=True,
    )
    parser.add_argument(
        "--km-path",
        type=pathlib.Path,
        help="Where to write the fitted MiniBatchKMeans model.",
        required=True,
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=100,
        help="Number of k-means clusters. (Default: 100)",
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=1.0,
        help="Fraction of frames to fit on. Use 0.1 for transformer features. (Default: 1.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="MiniBatchKMeans batch size. (Default: 10000)",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=20,
        help="k-means++ random starts. (Default: 20)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="MiniBatchKMeans max_iter. (Default: 100)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Cap frames used as the k-means pool. -1 = all frames. (Default: -1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. (Default: 0)",
    )
    args = parser.parse_args()
    run_learn(args)


if __name__ == "__main__":
    cli_main()
