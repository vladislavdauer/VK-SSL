import json
import pathlib
import time
from argparse import ArgumentParser

import joblib
import numpy as np

from src.models.hubert.kmeans import (
    default_fit_splits,
    fit_kmeans,
    sample_row_ranges,
    split_row_ranges,
)

def run_learn(args):
    features = np.load(args.features_path, mmap_mode="r")
    with open(args.index_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    print(
        f"features: {features.shape[0]} frames x {features.shape[1]} "
        f"({features.nbytes / 2**30:.2f} GiB, mmap from {args.features_path})",
        flush=True,
    )
    total_len = sum(int(item["length"]) for item in meta["index"])
    if total_len != int(features.shape[0]):
        raise ValueError(
            f"index.json lengths sum to {total_len} frames, but features.npy has "
            f"{features.shape[0]} rows — features and index come from different runs"
        )

    fit_splits = args.fit_splits or default_fit_splits(meta["index"])
    ranges = split_row_ranges(meta["index"], fit_splits)
    n_fit_rows = sum(stop - start for start, stop in ranges)
    print(f"fit splits: {fit_splits} ({n_fit_rows} frames)", flush=True)
    if n_fit_rows == 0:
        raise ValueError(f"No frames for --fit-splits {fit_splits}")

    started = time.time()
    fit_feats = sample_row_ranges(
        features,
        ranges,
        percent=float(args.percent),
        rng=np.random.RandomState(int(args.seed)),
        max_frames=int(args.max_frames),
        progress=True,
    )
    model = fit_kmeans(
        fit_feats,
        n_clusters=int(args.n_clusters),
        batch_size=int(args.batch_size),
        n_init=int(args.n_init),
        max_iter=int(args.max_iter),
        seed=int(args.seed),
        progress=True,
    )
    print(f"fit done in {(time.time() - started) / 60:.1f} min", flush=True)

    fit_stats = getattr(model, "_hubert_fit_stats", {})
    if fit_stats:
        print(
            f"quality: used={fit_stats.get('used_clusters')}/{args.n_clusters} "
            f"entropy={fit_stats.get('entropy'):.3f} "
            f"inertia/sample={fit_stats.get('inertia_per_sample'):.5f} "
            f"fit_frames={fit_stats.get('fit_frames')}"
        )
        if fit_stats.get("used_clusters", 0) < max(2, int(0.5 * args.n_clusters)):
            raise RuntimeError(
                "Too many empty clusters after k-means fit; check features / n_clusters."
            )

    args.km_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.km_path)

    info = {
        "n_clusters": int(args.n_clusters),
        "percent": float(args.percent),
        "batch_size": int(args.batch_size),
        "n_init": int(args.n_init),
        "max_iter": int(args.max_iter),
        "seed": int(args.seed),
        "max_frames": int(args.max_frames),
        "fit_splits": list(fit_splits),
        "feature_type": meta.get("feature_type"),
        "layer": meta.get("layer"),
        "label_rate": meta.get("label_rate"),
        "n_frames": int(features.shape[0]),
        "dim": int(features.shape[1]) if features.ndim == 2 else 0,
        "fit": fit_stats,
    }
    with open(args.km_path.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)

    print(f"saved {args.km_path}")

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
        "--fit-splits",
        nargs="+",
        default=None,
        help=(
            "Splits from index.json to fit on. Default: every split except dev-*/test-*, "
            "as fairseq fits the teacher on train only."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=50_000_000,
        help=(
            "Hard cap on frames used for the fit; they are copied into RAM. "
            "50M MFCC frames is ~140 h of audio and ~7.5 GiB. "
            "Use 0 to fit on everything (needs ~50 GiB RAM for 960 h). (Default: 50000000)"
        ),
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
        "--seed",
        type=int,
        default=0,
        help="Random seed. (Default: 0)",
    )
    args = parser.parse_args()
    run_learn(args)

if __name__ == "__main__":
    cli_main()
