import json
import pathlib
import struct
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm

from src.data.duration_cache import open_subset
from src.data.hubert_transforms import librispeech_utt_id, waveform_16k
from src.models.hubert.config import get_hubert_config
from src.models.hubert.hubert_model import HubertModel
from src.models.hubert.kmeans import extract_mfcc_39

class NpyStreamWriter:
\
\
\
\
\

    HEADER_BYTES = 4096
    MAGIC = b"\x93NUMPY\x01\x00"

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.handle = open(self.path, "wb")
        self.handle.write(b"\0" * self.HEADER_BYTES)
        self.rows = 0
        self.dim = None

    def append(self, rows: np.ndarray):
        rows = np.ascontiguousarray(rows, dtype="<f4")
        if rows.ndim != 2:
            raise ValueError(f"Expected 2-D features, got shape {rows.shape}")
        if self.dim is None:
            self.dim = int(rows.shape[1])
        elif rows.shape[1] != self.dim:
            raise ValueError(f"Feature dim changed from {self.dim} to {rows.shape[1]}")

        self.handle.write(rows.tobytes())
        self.rows += int(rows.shape[0])

    def close(self):
        header = repr(
            {"descr": "<f4", "fortran_order": False, "shape": (self.rows, self.dim or 0)}
        )
        body_len = self.HEADER_BYTES - len(self.MAGIC) - 2
        header = header.ljust(body_len - 1) + "\n"
        self.handle.seek(0)
        self.handle.write(self.MAGIC + struct.pack("<H", body_len) + header.encode("latin1"))
        self.handle.close()

    def abort(self):
        self.handle.close()
        self.path.unlink(missing_ok=True)

def _parse_urls(args):
    if args.sanity_check:
        return ["dev-clean"]
    return list(args.subsets)

def _load_encoder(args):
    num_classes = [int(v) for v in str(args.num_classes).split(",") if str(v).strip()]
    model = HubertModel(
        get_hubert_config(
            args.model_size,
            num_classes=num_classes,
            label_rate=float(args.label_rate),
        )
    )
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    cleaned = {}
    for key, value in state.items():
        if key.startswith("model."):
            cleaned[key[len("model.") :]] = value
        else:
            cleaned[key] = value

    model.load_state_dict(cleaned, strict=False)
    model.eval()
    if args.use_cuda and torch.cuda.is_available():
        model = model.cuda()

    return model

def extract_one(args, model, sample):
    wav = waveform_16k(sample, 16000)
    if args.feature_type == "mfcc":
        feats = extract_mfcc_39(wav, 16000)
        return feats.numpy()

    lengths = torch.tensor([wav.numel()], dtype=torch.long)
    source = wav.unsqueeze(0)
    if next(model.parameters()).is_cuda:
        source = source.cuda()
        lengths = lengths.cuda()

    layer = int(args.layer) - 1
    with torch.no_grad():
        hidden, feat_lengths, _ = model.extract_features(source, lengths, tgt_layer=layer)

    t = int(feat_lengths[0].item())
    return hidden[0, :t].detach().cpu().numpy()

def run_extract(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = None
    if args.feature_type == "hubert":
        if args.checkpoint_path is None:
            raise ValueError("--checkpoint-path is required for feature_type=hubert")

        model = _load_encoder(args)

    index = []
    writer = NpyStreamWriter(args.out_dir / "features.npy")
    try:
        for url in _parse_urls(args):
            dataset = open_subset(args.librispeech_path, url)
            n = len(dataset) if not args.sanity_check else min(len(dataset), 32)
            for i in tqdm(range(n), desc=f"extract/{url}"):
                sample = dataset[i]
                utt_id = librispeech_utt_id(sample)
                feat = extract_one(args, model, sample)
                writer.append(feat)
                index.append({"id": utt_id, "split": url, "length": int(feat.shape[0])})
    except BaseException:
        writer.abort()
        raise

    writer.close()

    with open(args.out_dir / "index.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "feature_type": args.feature_type,
                "layer": int(args.layer) if args.feature_type == "hubert" else None,
                "label_rate": 100 if args.feature_type == "mfcc" else 50,
                "index": index,
            },
            handle,
        )

def cli_main():
    parser = ArgumentParser()
    parser.add_argument(
        "--librispeech-path",
        type=pathlib.Path,
        help="Path to LibriSpeech datasets.",
        required=True,
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        help="Directory to write features.npy and index.json.",
        required=True,
    )
    parser.add_argument(
        "--feature-type",
        choices=["mfcc", "hubert"],
        default="mfcc",
        help="Teacher features: MFCC iter1 or HuBERT layer. (Default: mfcc)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        default=None,
        help="HuBERT checkpoint. Required when --feature-type is hubert.",
    )
    parser.add_argument(
        "--model-size",
        default="base",
        choices=["tiny", "base", "large", "xlarge"],
        help="HuBERT size. Must match the checkpoint. (Default: base)",
    )
    parser.add_argument(
        "--num-classes",
        default="100",
        help="Comma-separated codebook sizes used at pre-train. (Default: 100)",
    )
    parser.add_argument(
        "--label-rate",
        default=50.0,
        type=float,
        help="Label rate used to rebuild the encoder. (Default: 50)",
    )
    parser.add_argument(
        "--layer",
        default=6,
        type=int,
        help="1-based transformer layer to dump for --feature-type hubert. (Default: 6)",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["train-clean-100", "train-clean-360", "train-other-500"],
        help=(
            "Splits to extract. Include dev-clean dev-other so pre-train validation has "
            "labels. (Default: train-clean-100 train-clean-360 train-other-500)"
        ),
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Run HuBERT feature extraction on CUDA.",
    )
    parser.add_argument(
        "--sanity_check",
        action="store_true",
        help="Extract a small dev-clean subset only.",
    )
    args = parser.parse_args()
    run_extract(args)

if __name__ == "__main__":
    cli_main()
