from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torchaudio
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

_HAS_CONVERGENCE_HOOK = hasattr(MiniBatchKMeans, "_mini_batch_convergence")

class ProgressMiniBatchKMeans(MiniBatchKMeans):
    _bar = None

    def _mini_batch_convergence(
        self, step, n_steps, n_samples, centers_squared_diff, batch_inertia
    ):
        stop = super()._mini_batch_convergence(
            step, n_steps, n_samples, centers_squared_diff, batch_inertia
        )
        if self._bar is not None:
            self._bar.total = n_steps
            self._bar.update(step + 1 - self._bar.n)
            inertia = self._ewa_inertia
            if inertia is None:
                inertia = batch_inertia / self._batch_size

            self._bar.set_postfix(
                inertia=f"{inertia:.4f}",
                stale=f"{self._no_improvement}/{self.max_no_improvement}",
                refresh=False,
            )

        return stop

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_bar"] = None
        return state

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
) -> MiniBatchKMeans:
    cls = ProgressMiniBatchKMeans if _HAS_CONVERGENCE_HOOK else MiniBatchKMeans
    return cls(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=0,
        compute_labels=False,
        tol=0.0,
        max_no_improvement=100,
        n_init=n_init,
        reassignment_ratio=0.0,
        random_state=random_state,
    )

CHUNK_BYTES = 512 << 20

def _chunk_rows(features: np.ndarray) -> int:

    dim = int(features.shape[1]) if features.ndim == 2 else 1
    return max(1, CHUNK_BYTES // (4 * max(dim, 1)))

def sample_frames(
    features: np.ndarray,
    percent: float,
    rng: np.random.RandomState,
    max_frames: int = 0,
    chunk: int = 0,
    progress: bool = False,
) -> np.ndarray:
\
\
\
\

    n = int(features.shape[0])
    chunk = chunk or _chunk_rows(features)
    n_keep = n
    if 0 <= percent < 1.0:
        n_keep = max(1, int(np.ceil(n * percent)))
    if max_frames and max_frames > 0:
        n_keep = min(n_keep, int(max_frames))

    if n_keep >= n:
        if isinstance(features, np.memmap):
            return np.array(features, dtype=np.float32, order="C")

        return np.ascontiguousarray(features, dtype=np.float32)

    out = np.empty((n_keep, features.shape[1]), dtype=np.float32)
    pos = 0
    bar = tqdm(
        total=n,
        desc="sampling",
        unit="frame",
        unit_scale=True,
        disable=not progress,
    )
    with bar:
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            remaining_rows = n - start
            take = int(round((n_keep - pos) * (stop - start) / remaining_rows))
            take = min(take, stop - start, n_keep - pos)
            if take > 0:
                block = np.asarray(features[start:stop])
                sel = np.sort(rng.choice(block.shape[0], size=take, replace=False))
                out[pos : pos + take] = block[sel]
                pos += take

            bar.update(stop - start)
            bar.set_postfix(kept=f"{pos}/{n_keep}", refresh=False)

    return out[:pos]

def default_fit_splits(index: Sequence[dict]) -> List[str]:

    splits = list(dict.fromkeys(item["split"] for item in index))
    train = [s for s in splits if not s.startswith(("dev", "test"))]
    return train or splits

def split_row_ranges(index: Sequence[dict], splits: Sequence[str]) -> List[Tuple[int, int]]:

    wanted = set(splits)
    ranges = []
    offset = 0
    for item in index:
        length = int(item["length"])
        if item["split"] in wanted and length > 0:
            if ranges and ranges[-1][1] == offset:
                ranges[-1] = (ranges[-1][0], offset + length)
            else:
                ranges.append((offset, offset + length))

        offset += length

    return ranges

def sample_row_ranges(
    features: np.ndarray,
    ranges: Sequence[Tuple[int, int]],
    percent: float,
    rng: np.random.RandomState,
    max_frames: int = 0,
    progress: bool = False,
) -> np.ndarray:
    total = sum(stop - start for start, stop in ranges)
    if total == 0:
        return np.zeros((0, features.shape[1]), dtype=np.float32)

    n_keep = total
    if 0 <= percent < 1.0:
        n_keep = max(1, int(np.ceil(total * percent)))
    if max_frames and max_frames > 0:
        n_keep = min(n_keep, int(max_frames))

    parts = []
    for start, stop in ranges:
        share = max(1, int(round(n_keep * (stop - start) / total)))
        parts.append(
            sample_frames(features[start:stop], 1.0, rng, max_frames=share, progress=progress)
        )

    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

def cluster_usage_stats(labels: np.ndarray, n_clusters: int) -> dict:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.size == 0:
        return {
            "n_labels": 0,
            "used_clusters": 0,
            "empty_clusters": int(n_clusters),
            "entropy": 0.0,
            "min_count": 0,
            "max_count": 0,
        }

    counts = np.bincount(labels, minlength=int(n_clusters)).astype(np.float64)
    used = int((counts > 0).sum())
    probs = counts / counts.sum()
    nz = probs[probs > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    return {
        "n_labels": int(labels.size),
        "used_clusters": used,
        "empty_clusters": int(n_clusters) - used,
        "entropy": entropy,
        "min_count": int(counts.min()),
        "max_count": int(counts.max()),
    }

def fit_kmeans(
    features: np.ndarray,
    n_clusters: int,
    percent: float = 1.0,
    batch_size: int = 10000,
    n_init: int = 20,
    max_iter: int = 100,
    seed: int = 0,
    max_frames: int = 0,
    progress: bool = False,
) -> MiniBatchKMeans:
    rng = np.random.RandomState(seed)
    sampled = sample_frames(features, percent, rng, max_frames=max_frames, progress=progress)
    if sampled.shape[0] < int(n_clusters):
        raise ValueError(
            f"Need at least n_clusters={n_clusters} frames to fit k-means, got {sampled.shape[0]}"
        )

    model = build_kmeans(
        n_clusters=n_clusters,
        batch_size=min(int(batch_size), int(sampled.shape[0])),
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
    )
    bar = tqdm(desc=f"kmeans/{n_clusters}", unit="step", disable=not progress)
    with bar:
        if isinstance(model, ProgressMiniBatchKMeans):
            model._bar = bar
        try:
            model.fit(sampled)
        finally:
            if isinstance(model, ProgressMiniBatchKMeans):
                model._bar = None

    eval_n = min(int(sampled.shape[0]), 200_000)
    if eval_n < sampled.shape[0]:
        eval_idx = np.sort(rng.choice(sampled.shape[0], size=eval_n, replace=False))
        eval_feats = np.ascontiguousarray(sampled[eval_idx], dtype=np.float32)
    else:
        eval_feats = sampled

    pred = model.predict(eval_feats)
    usage = cluster_usage_stats(pred, n_clusters)
    inertia = float(-model.score(eval_feats) / max(len(eval_feats), 1))
    model._hubert_fit_stats = {
        "inertia_per_sample": inertia,
        "fit_frames": int(sampled.shape[0]),
        "eval_frames": int(eval_feats.shape[0]),
        **usage,
    }
    return model

def predict_labels(
    kmeans: MiniBatchKMeans,
    features: np.ndarray,
    chunk: int = 0,
    progress: bool = False,
) -> np.ndarray:
    n = int(features.shape[0])
    chunk = chunk or _chunk_rows(features)
    out = np.empty(n, dtype=np.int32)
    bar = tqdm(
        total=n,
        desc="predict",
        unit="frame",
        unit_scale=True,
        disable=not progress,
    )
    with bar:
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            block = np.ascontiguousarray(features[start:stop], dtype=np.float32)
            out[start:stop] = kmeans.predict(block)
            bar.update(stop - start)

    return out

def concat_utterance_features(frames: Sequence[np.ndarray]) -> np.ndarray:
    if not frames:
        return np.zeros((0, 0), dtype=np.float32)

    return np.concatenate([np.asarray(item, dtype=np.float32) for item in frames], axis=0)

def split_labels(labels: Iterable[int], lengths: Sequence[int]) -> List[List[int]]:
    labels = list(labels)
    out = []
    offset = 0
    for length in lengths:
        out.append([int(v) for v in labels[offset : offset + length]])
        offset += length

    return out
