from typing import List, Sequence

from src.data.hubert_transforms import (
    DummyHubertPretrainTransform,
    HubertFinetuneTestTransform,
    HubertFinetuneTransform,
    HubertPretrainTransform,
    load_km_file,
)
from src.data.librispeech_data_module import LibriSpeechDataModule


def _load_label_maps(label_paths: Sequence[str]) -> List[dict]:
    maps = []
    for path in label_paths:
        loaded = load_km_file(path)
        if not loaded:
            raise ValueError(f"Empty HuBERT label file: {path}")

        maps.append(loaded)

    return maps


def get_hubert_pretrain_data_module(
    librispeech_path,
    label_paths=None,
    dummy_labels=False,
    num_classes=None,
    label_rate=50.0,
    sanity_check=False,
    durations_cache_dir=None,
    num_workers=4,
    max_batch_duration=87.5,
    min_duration=0.1,
    max_duration=20.0,
):
    if dummy_labels:
        if num_classes is None:
            num_classes = [100]
        transform = DummyHubertPretrainTransform(num_classes=num_classes, label_rate=label_rate)
    else:
        if not label_paths:
            raise ValueError("label_paths is required unless dummy_labels=True")
        transform = HubertPretrainTransform(_load_label_maps(label_paths))

    return LibriSpeechDataModule(
        librispeech_path=librispeech_path,
        train_transform=transform,
        val_transform=transform,
        test_transform=transform,
        batch_size=None,
        min_duration=min_duration,
        max_duration=max_duration,
        max_batch_duration=max_batch_duration,
        train_num_buckets=50,
        train_shuffle=True,
        num_workers=num_workers,
        durations_cache_dir=durations_cache_dir,
        sanity_check=sanity_check,
    )


def get_hubert_finetune_data_module(
    librispeech_path,
    sp_model_path,
    sanity_check=False,
    durations_cache_dir=None,
    num_workers=4,
    max_batch_duration=200.0,
    min_duration=0.1,
    max_duration=20.0,
):
    train_transform = HubertFinetuneTransform(sp_model_path)
    val_transform = HubertFinetuneTransform(sp_model_path)
    test_transform = HubertFinetuneTestTransform(sp_model_path)

    return LibriSpeechDataModule(
        librispeech_path=librispeech_path,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        batch_size=None,
        min_duration=min_duration,
        max_duration=max_duration,
        max_batch_duration=max_batch_duration,
        train_num_buckets=50,
        train_shuffle=True,
        num_workers=num_workers,
        durations_cache_dir=durations_cache_dir,
        sanity_check=sanity_check,
    )
