from typing import Iterable, List, Sequence

import numpy as np
import torch
import torchaudio
from sklearn.cluster import MiniBatchKMeans


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
    return MiniBatchKMeans(
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


def sample_frames(features: np.ndarray, percent: float, rng: np.random.RandomState) -> np.ndarray:
    if percent >= 1.0 or percent < 0:
        return features

    n = features.shape[0]
    n_keep = max(1, int(np.ceil(n * percent)))
    idx = rng.choice(n, size=n_keep, replace=False)
    return features[idx]


def fit_kmeans(
    features: np.ndarray,
    n_clusters: int,
    percent: float = 1.0,
    batch_size: int = 10000,
    n_init: int = 20,
    max_iter: int = 100,
    seed: int = 0,
) -> MiniBatchKMeans:
    rng = np.random.RandomState(seed)
    sampled = sample_frames(np.asarray(features, dtype=np.float32), percent, rng)
    model = build_kmeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
    )
    model.fit(sampled)
    return model


def predict_labels(kmeans: MiniBatchKMeans, features: np.ndarray) -> np.ndarray:
    return kmeans.predict(np.asarray(features, dtype=np.float32))


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
