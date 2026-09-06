import os
import random
from pathlib import Path

import torch
import torchaudio
from pytorch_lightning import LightningDataModule

from src.data.data_transforms import (
    TrainTransform, ValTransform, TestTransform
)
from src.data.duration_cache import load_durations


def _batch_by_length(idx_lengths, batch_size, max_batch_duration=None):
    batches = []
    current_batch = []
    current_duration = 0.0
    for idx, duration in idx_lengths:
        would_exceed_duration = (
            max_batch_duration is not None
            and current_batch
            and current_duration + duration > max_batch_duration
        )
        would_exceed_count = batch_size is not None and len(current_batch) == batch_size
        if would_exceed_duration or would_exceed_count:
            batches.append(current_batch)
            current_batch = [idx]
            current_duration = duration
        else:
            current_batch.append(idx)
            current_duration += duration

    if current_batch:
        batches.append(current_batch)

    return batches


def filter_by_duration(dataset, durations, min_duration, max_duration):
    keep_idx = [
        i
        for i, dur in enumerate(durations)
        if min_duration <= dur <= max_duration
    ]
    if not keep_idx:
        raise RuntimeError(
            f"No samples left after duration filter "
            f"[{min_duration}, {max_duration}] s "
            f"(had {len(durations)} samples)."
        )
    filtered_durations = [durations[i] for i in keep_idx]
    return torch.utils.data.Subset(dataset, keep_idx), filtered_durations


class CustomBucketDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        lengths,
        num_buckets,
        shuffle=False,
        batch_size=None,
        max_batch_duration=None,
    ):
        super().__init__()

        assert len(dataset) == len(lengths)

        self.dataset = dataset

        max_length = max(lengths)
        min_length = min(lengths)

        if max_batch_duration is not None:
            assert max_batch_duration >= max_length

        if num_buckets <= 1 or max_length == min_length:
            bucket_assignments = torch.zeros(len(lengths), dtype=torch.long)
        else:
            buckets = torch.linspace(min_length, max_length, num_buckets)
            bucket_assignments = torch.bucketize(torch.tensor(lengths), buckets)

        idx_length_buckets = [
            (idx, length, int(bucket_assignments[idx]))
            for idx, length in enumerate(lengths)
        ]
        if shuffle:
            idx_length_buckets = random.sample(idx_length_buckets, len(idx_length_buckets))
        else:
            idx_length_buckets = sorted(idx_length_buckets, key=lambda x: x[1], reverse=True)

        sorted_idx_length_buckets = sorted(idx_length_buckets, key=lambda x: x[2])
        self.batches = _batch_by_length(
            [(idx, length) for idx, length, _ in sorted_idx_length_buckets],
            batch_size=batch_size,
            max_batch_duration=max_batch_duration,
        )

    def __getitem__(self, idx):
        return [self.dataset[subidx] for subidx in self.batches[idx]]

    def __len__(self):
        return len(self.batches)


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform_fn):
        self.dataset = dataset
        self.transform_fn = transform_fn

    def __getitem__(self, idx):
        return self.transform_fn(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)


DEFAULT_TRAIN_SUBSETS = (
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
)


class LibriSpeechDataModule(LightningDataModule):
    librispeech_cls = torchaudio.datasets.LIBRISPEECH

    def __init__(
        self,
        *,
        librispeech_path,
        train_transform,
        val_transform,
        test_transform,
        batch_size=64,
        min_duration=0.1,
        max_duration=16.7,
        max_batch_duration=None,
        train_num_buckets=50,
        train_shuffle=True,
        num_workers=4,
        durations_cache_dir=None,
        sanity_check=False,
        train_subsets=None,
    ):
        super().__init__()
        self.librispeech_path = librispeech_path
        self.durations_cache_dir = Path(
            durations_cache_dir
            if durations_cache_dir is not None
            else Path(librispeech_path) / ".duration_cache"
        )
        self.train_dataset_durations = None
        self.val_dataset_durations = None
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_batch_duration = max_batch_duration
        self.train_num_buckets = train_num_buckets
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers
        self.sanity_check = sanity_check
        self.train_subsets = list(train_subsets) if train_subsets else list(DEFAULT_TRAIN_SUBSETS)

    def _prepare_datasets(self, urls, durations_cache_attr, num_buckets):
        datasets = [
            self.librispeech_cls(self.librispeech_path, url=url) for url in urls
        ]

        cached = getattr(self, durations_cache_attr)
        if not cached:
            cached = []
            for url, dataset in zip(urls, datasets):
                durations = load_durations(
                    self.durations_cache_dir,
                    url,
                    expected_n=len(dataset._walker),
                )
                if int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0))) == 0:
                    print(
                        f"Loaded duration cache: "
                        f"{self.durations_cache_dir / (url.replace('-', '_') + '_durations.json')} "
                        f"({len(durations)})",
                        flush=True,
                    )
                cached.append(durations)
            setattr(self, durations_cache_attr, cached)

        bucketed = []
        for dataset, durations in zip(datasets, cached):
            filtered_ds, filtered_durs = filter_by_duration(
                dataset,
                durations,
                self.min_duration,
                self.max_duration,
            )
            bucketed.append(
                CustomBucketDataset(
                    filtered_ds,
                    filtered_durs,
                    num_buckets=num_buckets,
                    shuffle=False,
                    batch_size=self.batch_size,
                    max_batch_duration=self.max_batch_duration,
                )
            )
        return torch.utils.data.ConcatDataset(bucketed)

    def train_dataloader(self):
        if self.sanity_check:
            urls = ["dev-clean"]
        else:
            urls = self.train_subsets

        dataset = self._prepare_datasets(
            urls,
            "train_dataset_durations",
            num_buckets=self.train_num_buckets,
        )
        dataset = TransformDataset(dataset, self.train_transform)
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=None,
            shuffle=self.train_shuffle,
        )

    def val_dataloader(self):
        if self.sanity_check:
            urls = ["dev-clean"]
        else:
            urls = ["dev-clean", "dev-other"]

        dataset = self._prepare_datasets(
            urls,
            "val_dataset_durations",
            num_buckets=1,
        )
        dataset = TransformDataset(dataset, self.val_transform)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        if self.sanity_check:
            dataset = self.librispeech_cls(self.librispeech_path, url="dev-clean")
        else:
            dataset = self.librispeech_cls(self.librispeech_path, url="test-clean")
        dataset = TransformDataset(dataset, self.test_transform)
        return torch.utils.data.DataLoader(dataset, batch_size=None)


def get_data_module(
        librispeech_path,
        global_stats_path,
        sp_model_path,
        sanity_check=False,
        durations_cache_dir=None,
        num_workers=4,
        train_subsets=None,
        ):
    train_transform = TrainTransform(
        global_stats_path=global_stats_path, sp_model_path=sp_model_path
        )
    val_transform = ValTransform(
        global_stats_path=global_stats_path, sp_model_path=sp_model_path
        )
    test_transform = TestTransform(
        global_stats_path=global_stats_path, sp_model_path=sp_model_path
        )
    return LibriSpeechDataModule(
        librispeech_path=librispeech_path,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        sanity_check=sanity_check,
        durations_cache_dir=durations_cache_dir,
        num_workers=num_workers,
        min_duration=0.1,
        max_duration=16.7,
        train_subsets=train_subsets,
    )
