import json
import pathlib
from argparse import ArgumentParser

import joblib
import numpy as np
from tqdm import tqdm

from src.models.hubert.kmeans import cluster_usage_stats, predict_labels, split_labels


def run_dump(args):
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

    lengths = [int(item["length"]) for item in meta["index"]]
    total_len = int(sum(lengths))
    if total_len != int(features.shape[0]):
        raise ValueError(
            f"index.json lengths sum to {total_len} frames, but features.npy has "
            f"{features.shape[0]} rows — refuse to dump misaligned labels"
        )

    with tqdm(total=1, desc="load km model", unit="file") as bar:
        kmeans = joblib.load(args.km_path)
        bar.update(1)

    n_clusters = int(getattr(kmeans, "n_clusters", args.n_clusters))
    labels = predict_labels(kmeans, features, show_progress=True)
    if labels.min() < 0 or labels.max() >= n_clusters:
        raise ValueError(
            f"Predicted labels out of range: min={labels.min()} max={labels.max()} "
            f"n_clusters={n_clusters}"
        )

    usage = cluster_usage_stats(labels, n_clusters)
    tqdm.write(
        f"label usage: used={usage['used_clusters']}/{n_clusters} "
        f"empty={usage['empty_clusters']} entropy={usage['entropy']:.3f} "
        f"min/max count={usage['min_count']}/{usage['max_count']}"
    )
    if usage["used_clusters"] < max(2, int(0.5 * n_clusters)):
        raise RuntimeError(
            "Too many empty clusters in teacher labels; re-run learn_kmeans."
        )

    per_utt = split_labels(labels, lengths)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_path, "w", encoding="utf-8") as handle:
        for item, seq in tqdm(
            zip(meta["index"], per_utt),
            total=len(per_utt),
            desc="write .km",
            unit="utt",
        ):
            if len(seq) != int(item["length"]):
                raise ValueError(
                    f"Utterance {item['id']}: wrote {len(seq)} labels, "
                    f"index expects {item['length']}"
                )
            handle.write(item["id"] + " " + " ".join(str(v) for v in seq) + "\n")

    sidecar = {
        "n_clusters": n_clusters,
        "label_rate": meta.get("label_rate"),
        "feature_type": meta.get("feature_type"),
        "layer": meta.get("layer"),
        "n_utterances": len(per_utt),
        "n_frames": int(labels.shape[0]),
        "usage": usage,
    }
    with open(args.out_path.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2)

    tqdm.write(f"saved {args.out_path}")


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
        help="Fitted MiniBatchKMeans model from learn_kmeans.py.",
        required=True,
    )
    parser.add_argument(
        "--out-path",
        type=pathlib.Path,
        help="Output .km file with 'utt_id z1 z2 ...' lines.",
        required=True,
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=100,
        help="Fallback cluster count written to sidecar json. (Default: 100)",
    )
    args = parser.parse_args()
    run_dump(args)


if __name__ == "__main__":
    cli_main()
