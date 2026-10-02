from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Sequence

import torchaudio
from tqdm import tqdm

LIBRILIGHT_SUBSETS = {"ll-10min": "10min", "ll-1h": "1h", "ll-10h": "10h"}

def open_subset(root: Path | str, name: str):
    if name in LIBRILIGHT_SUBSETS:
        return torchaudio.datasets.LibriLightLimited(str(root), subset=LIBRILIGHT_SUBSETS[name])

    return torchaudio.datasets.LIBRISPEECH(str(root), url=name)

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

def dataset_audio_paths(dataset) -> List[str]:
    if isinstance(dataset, torchaudio.datasets.LibriLightLimited):
        paths = []
        for folder, fileid in dataset._fileids_paths:
            speaker_id, chapter_id, _ = fileid.split("-")
            paths.append(
                str(
                    Path(dataset._path)
                    / folder
                    / speaker_id
                    / chapter_id
                    / f"{fileid}{dataset._ext_audio}"
                )
            )
        return paths

    return [audio_path(dataset, fileid) for fileid in dataset._walker]

def _one_duration(path: str) -> float:
    info = torchaudio.info(path)
    return info.num_frames / float(info.sample_rate)

def scan_sample_durations(
    librispeech_dataset,
    desc: str = "durations",
    num_workers: int = 32,
) -> List[float]:
    paths = dataset_audio_paths(librispeech_dataset)

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
        ds = open_subset(librispeech_path, url)
        n = len(ds)
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
        "100h": ["train-clean-100", "dev-clean", "dev-other"],
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
        *LIBRILIGHT_SUBSETS,
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
        nargs="+",
        choices=list(known_groups) + single_urls,
        default=["all"],
        help=(
            "Groups (train/val/100h/all) and/or single splits, e.g. "
            "'--subset 100h ll-10h'. (Default: all)"
        ),
    )
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=32,
        help="Parallel torchaudio.info threads. (Default: 32)",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or (args.librispeech_path / ".duration_cache")
    urls = []
    for name in args.subset:
        for url in known_groups.get(name, [name]):
            if url not in urls:
                urls.append(url)

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
