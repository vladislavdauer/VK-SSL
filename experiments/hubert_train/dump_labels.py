import json
import pathlib
from argparse import ArgumentParser

import joblib
import numpy as np

from src.models.hubert.kmeans import predict_labels, split_labels


def run_dump(args):
    features = np.load(args.features_path)
    with open(args.index_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    kmeans = joblib.load(args.km_path)
    labels = predict_labels(kmeans, features)
    lengths = [int(item["length"]) for item in meta["index"]]
    per_utt = split_labels(labels, lengths)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as handle:
        for item, seq in zip(meta["index"], per_utt):
            handle.write(item["id"] + " " + " ".join(str(v) for v in seq) + "\n")

    sidecar = {
        "n_clusters": int(getattr(kmeans, "n_clusters", args.n_clusters)),
        "label_rate": meta.get("label_rate"),
        "feature_type": meta.get("feature_type"),
        "layer": meta.get("layer"),
        "n_utterances": len(per_utt),
    }
    with open(args.out_path.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle)


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
