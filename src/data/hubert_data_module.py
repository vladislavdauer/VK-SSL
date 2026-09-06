from typing import List, Sequence

from src.data.hubert_transforms import (
    DummyHubertPretrainTransform,
    HubertCharFinetuneTestTransform,
    HubertCharFinetuneTransform,
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
    min_duration=2.0,
    max_duration=60.0,
    max_sample_size=250000,
    random_crop=True,
    train_subsets=None,
):
    if dummy_labels:
        if num_classes is None:
            num_classes = [100]
        transform = DummyHubertPretrainTransform(num_classes=num_classes, label_rate=label_rate)
    else:
        if not label_paths:
            raise ValueError("label_paths is required unless dummy_labels=True")
        transform = HubertPretrainTransform(
            _load_label_maps(label_paths),
            label_rate=float(label_rate),
            max_sample_size=int(max_sample_size),
            random_crop=bool(random_crop),
        )

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
        train_subsets=train_subsets,
    )


def get_hubert_finetune_data_module(
    librispeech_path,
    sp_model_path=None,
    label_type="char",
    sanity_check=False,
    durations_cache_dir=None,
    num_workers=4,
    batch_size=None,
    max_batch_duration=200.0,
    min_duration=0.1,
    max_duration=20.0,
    train_subsets=None,
):
    if label_type == "spm":
        if not sp_model_path:
            raise ValueError("sp_model_path is required when label_type='spm'")
        train_transform = HubertFinetuneTransform(sp_model_path)
        val_transform = HubertFinetuneTransform(sp_model_path)
        test_transform = HubertFinetuneTestTransform(sp_model_path)
    else:
        train_transform = HubertCharFinetuneTransform()
        val_transform = HubertCharFinetuneTransform()
        test_transform = HubertCharFinetuneTestTransform()

    return LibriSpeechDataModule(
        librispeech_path=librispeech_path,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        batch_size=batch_size,
        min_duration=min_duration,
        max_duration=max_duration,
        max_batch_duration=max_batch_duration,
        train_num_buckets=50,
        train_shuffle=True,
        num_workers=num_workers,
        durations_cache_dir=durations_cache_dir,
        sanity_check=sanity_check,
        train_subsets=train_subsets,
    )
