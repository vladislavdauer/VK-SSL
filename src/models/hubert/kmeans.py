from typing import Iterable, List, Sequence

import numpy as np
import torch
import torchaudio


def extract_mfcc_39(waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    wav = waveform.detach().float().cpu().view(1, -1)
    mfcc = torchaudio.compliance.kaldi.mfcc(
        waveform=wav,
        sample_frequency=sample_rate,
        use_energy=False,
    )
    mfcc = mfcc.transpose(0, 1)
    delta = torchaudio.functional.compute_deltas(mfcc)
    ddelta = torchaudio.functional.compute_deltas(delta)
    concat = torch.cat([mfcc, delta, ddelta], dim=0)
    return concat.transpose(0, 1).contiguous()


def build_kmeans(
    n_clusters: int,
    batch_size: int = 10000,
    n_init: int = 20,
    max_iter: int = 100,
    random_state: int = 0,
    verbose: int = 0,
    tol: float = 0.0,
    max_no_improvement: int = 100,
    reassignment_ratio: float = 0.0,
):
    from sklearn.cluster import MiniBatchKMeans

    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=verbose,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
        random_state=random_state,
    )


def sample_frames(
    features: np.ndarray,
    percent: float,
    rng: np.random.RandomState,
    max_frames: int | None = None,
    show_progress: bool = False,
) -> np.ndarray:
    features = np.asarray(features)
    n = int(features.shape[0])
    n_keep = n
    if 0.0 <= percent < 1.0:
        n_keep = max(1, int(np.ceil(n * percent)))

    if max_frames is not None:
        n_keep = min(n_keep, max(1, int(max_frames)))

    if n_keep >= n:
        return features

    if show_progress:
        from tqdm import tqdm

        tqdm.write(f"sampling {n_keep}/{n} frames for k-means fit")

    idx = rng.choice(n, size=n_keep, replace=False)
    return np.asarray(features[idx], dtype=np.float32)


def cluster_usage_stats(labels: np.ndarray, n_clusters: int) -> dict:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.size == 0:
        return {
            "n_labels": 0,
            "n_clusters": int(n_clusters),
            "used_clusters": 0,
            "empty_clusters": int(n_clusters),
            "min_count": 0,
            "max_count": 0,
            "entropy": 0.0,
        }

    counts = np.bincount(labels, minlength=int(n_clusters)).astype(np.float64)
    used = int((counts > 0).sum())
    probs = counts / counts.sum()
    nz = probs[probs > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    return {
        "n_labels": int(labels.size),
        "n_clusters": int(n_clusters),
        "used_clusters": used,
        "empty_clusters": int(n_clusters) - used,
        "min_count": int(counts.min()),
        "max_count": int(counts.max()),
        "entropy": entropy,
    }


def fit_kmeans(
    features: np.ndarray,
    n_clusters: int,
    percent: float = 1.0,
    batch_size: int = 10000,
    n_init: int = 20,
    max_iter: int = 100,
    seed: int = 0,
    max_frames: int | None = None,
    show_progress: bool = False,
):
    from tqdm import tqdm

    rng = np.random.RandomState(seed)
    sampled = sample_frames(
        features,
        percent,
        rng,
        max_frames=max_frames,
        show_progress=show_progress,
    )
    n_samples = int(sampled.shape[0])
    if n_samples < int(n_clusters):
        raise ValueError(
            f"Need at least n_clusters={n_clusters} frames to fit k-means, got {n_samples}"
        )

    eff_batch = min(int(batch_size), n_samples)
    verbose = 1 if show_progress else 0
    if show_progress:
        tqdm.write(
            f"fitting MiniBatchKMeans.fit: {n_samples} frames, "
            f"C={n_clusters}, n_init={n_init}, max_iter={max_iter}, batch={eff_batch}"
        )

    model = build_kmeans(
        n_clusters=n_clusters,
        batch_size=eff_batch,
        n_init=max(1, int(n_init)),
        max_iter=max(1, int(max_iter)),
        random_state=seed,
        verbose=verbose,
    )
    model.fit(sampled)

    eval_size = min(n_samples, 200_000)
    if eval_size < n_samples:
        eval_idx = rng.choice(n_samples, size=eval_size, replace=False)
        eval_feats = np.asarray(sampled[eval_idx], dtype=np.float32)
    else:
        eval_feats = np.asarray(sampled, dtype=np.float32)

    inertia = float(-model.score(eval_feats) / max(len(eval_feats), 1))
    pred = model.predict(eval_feats)
    usage = cluster_usage_stats(pred, n_clusters)
    model._hubert_fit_stats = {
        "inertia_per_sample": inertia,
        "eval_frames": int(eval_feats.shape[0]),
        "fit_frames": n_samples,
        **usage,
    }
    if show_progress:
        tqdm.write(
            f"done: inertia/sample={inertia:.5f}, "
            f"used_clusters={usage['used_clusters']}/{n_clusters}, "
            f"entropy={usage['entropy']:.3f}"
        )
        if usage["used_clusters"] < max(2, int(0.5 * n_clusters)):
            tqdm.write(
                "WARNING: many empty clusters; re-fit with more frames / higher max_iter."
            )

    return model


def predict_labels(
    kmeans,
    features: np.ndarray,
    chunk_size: int = 1_000_000,
    show_progress: bool = False,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    n = features.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    if n <= chunk_size:
        return kmeans.predict(features).astype(np.int64, copy=False)

    from tqdm import tqdm

    labels = np.empty(n, dtype=np.int64)
    ranges = range(0, n, chunk_size)
    if show_progress:
        ranges = tqdm(ranges, desc="predict labels", unit="chunk")
    for start in ranges:
        end = min(start + chunk_size, n)
        labels[start:end] = kmeans.predict(features[start:end])

    return labels


def concat_utterance_features(frames: Sequence[np.ndarray]) -> np.ndarray:
    if not frames:
        return np.zeros((0, 0), dtype=np.float32)

    return np.concatenate([np.asarray(item, dtype=np.float32) for item in frames], axis=0)


def split_labels(labels: Iterable[int], lengths: Sequence[int]) -> List[List[int]]:
    labels = list(labels)
    out = []
    offset = 0
    for length in lengths:
        length = int(length)
        if length < 0:
            raise ValueError(f"Negative label length: {length}")
        out.append([int(v) for v in labels[offset : offset + length]])
        offset += length

    if offset != len(labels):
        raise ValueError(
            f"Label length mismatch: sum(lengths)={offset} != n_labels={len(labels)}"
        )

    return out
