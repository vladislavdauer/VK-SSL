from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Sequence

import torchaudio
from tqdm import tqdm


def duration_cache_path(cache_dir: Path | str, url: str) -> Path:
    return Path(cache_dir) / f"{url.replace('-', '_')}_durations.json"


def audio_path(librispeech_dataset, fileid: str) -> str:
    speaker_id, chapter_id, _ = fileid.split("-")
    return str(
        Path(librispeech_dataset._path)
        / speaker_id
        / chapter_id
        / f"{fileid}{librispeech_dataset._ext_audio}"
    )


def _one_duration(path: str) -> float:
    info = torchaudio.info(path)
    return info.num_frames / float(info.sample_rate)


def scan_sample_durations(
    librispeech_dataset,
    desc: str = "durations",
    num_workers: int = 32,
) -> List[float]:
    walker = list(librispeech_dataset._walker)
    paths = [audio_path(librispeech_dataset, fileid) for fileid in walker]

    if num_workers <= 1:
        return [
            _one_duration(p)
            for p in tqdm(paths, desc=desc, unit="utt", mininterval=1.0)
        ]

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        mapped = pool.map(_one_duration, paths, chunksize=64)
        return list(
            tqdm(mapped, total=len(paths), desc=desc, unit="utt", mininterval=1.0)
        )


def load_durations(cache_dir: Path | str, url: str, expected_n: int) -> List[float]:
    cache_file = duration_cache_path(cache_dir, url)
    if not cache_file.is_file():
        raise FileNotFoundError(
            f"Missing duration cache: {cache_file}\n"
            f"Build it once with:\n"
            f"  PYTHONPATH=. python -m src.data.duration_cache "
            f"--librispeech-path <path> --subset all"
        )
    with cache_file.open("r", encoding="utf-8") as f:
        durations = json.load(f)
    if len(durations) != expected_n:
        raise ValueError(
            f"Duration cache size mismatch for {url}: "
            f"cache has {len(durations)}, dataset has {expected_n}. "
            f"Rebuild with: python -m src.data.duration_cache"
        )
    return durations


def save_durations(cache_dir: Path | str, url: str, durations: Sequence[float]) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = duration_cache_path(cache_dir, url)
    tmp = cache_file.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(list(durations), f)
    tmp.replace(cache_file)
    return cache_file


def build_duration_caches(
    librispeech_path: str,
    urls: Sequence[str],
    cache_dir: str,
    scan_workers: int = 32,
) -> None:
    cache_dir = Path(cache_dir)
    for url in urls:
        ds = torchaudio.datasets.LIBRISPEECH(librispeech_path, url=url)
        n = len(ds._walker)
        print(f"Scanning {url} ({n} files)...", flush=True)
        durations = scan_sample_durations(
            ds, desc=f"durations/{url}", num_workers=scan_workers
        )
        out = save_durations(cache_dir, url, durations)
        print(f"Saved {out}", flush=True)


def main() -> None:
    from argparse import ArgumentParser

    known_groups = {
        "train": ["train-clean-100", "train-clean-360", "train-other-500"],
        "val": ["dev-clean", "dev-other"],
        "all": [
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
            "dev-clean",
            "dev-other",
        ],
    }
    single_urls = [
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
    ]

    parser = ArgumentParser(description="Build LibriSpeech duration JSON caches once.")
    parser.add_argument(
        "--librispeech-path",
        type=Path,
        required=True,
        help="Root LibriSpeech directory.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache dir (default: <librispeech>/.duration_cache).",
    )
    parser.add_argument(
        "--subset",
        choices=list(known_groups) + single_urls,
        default="all",
        help="Group (train/val/all) or a single split, e.g. train-other-500.",
    )
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=32,
        help="Parallel torchaudio.info threads. (Default: 32)",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or (args.librispeech_path / ".duration_cache")
    urls = known_groups.get(args.subset, [args.subset])

    print(f"Writing caches to {cache_dir}")
    build_duration_caches(
        str(args.librispeech_path),
        urls,
        str(cache_dir),
        scan_workers=args.scan_workers,
    )
    print("Done.")


if __name__ == "__main__":
    main()
