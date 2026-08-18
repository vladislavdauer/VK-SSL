import json
import pathlib
from argparse import ArgumentParser

import joblib
import numpy as np

from src.models.hubert.kmeans import fit_kmeans


def run_learn(args):
    features = np.load(args.features_path)
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
    )
    args.km_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.km_path)

    info = {
        "n_clusters": int(args.n_clusters),
        "percent": float(args.percent),
        "feature_type": meta.get("feature_type"),
        "layer": meta.get("layer"),
        "label_rate": meta.get("label_rate"),
        "n_frames": int(features.shape[0]),
        "dim": int(features.shape[1]) if features.ndim == 2 else 0,
    }
    with open(args.km_path.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(info, handle)


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
        help="Maximum k-means iterations. (Default: 100)",
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
