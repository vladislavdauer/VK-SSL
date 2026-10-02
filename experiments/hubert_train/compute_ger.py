import json
import pathlib
from argparse import ArgumentParser

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from src.data.hubert_transforms import librispeech_utt_id, waveform_16k
from src.models.hubert.config import get_hubert_config
from src.models.hubert.hubert_model import HubertModel

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
        elif key.startswith("encoder."):
            cleaned[key[len("encoder.") :]] = value
        else:
            cleaned[key] = value

    model.load_state_dict(cleaned, strict=False)
    model.eval()
    if args.use_cuda and torch.cuda.is_available():
        model = model.cuda()

    return model

def _extract_hidden(model, sample, layer_1based: int):
    wav = waveform_16k(sample, 16000)
    lengths = torch.tensor([wav.numel()], dtype=torch.long)
    source = wav.unsqueeze(0)
    if next(model.parameters()).is_cuda:
        source = source.cuda()
        lengths = lengths.cuda()

    layer = int(layer_1based) - 1
    with torch.no_grad():
        hidden, feat_lengths, _ = model.extract_features(source, lengths, tgt_layer=layer)

    t = int(feat_lengths[0].item())
    return hidden[0, :t].detach().float().cpu()

def effective_rank(matrix: torch.Tensor, eps: float = 1e-12) -> float:

    if matrix.ndim != 2:
        raise ValueError(f"expected 2D matrix, got {tuple(matrix.shape)}")

    if matrix.numel() == 0:
        return 0.0

    x = matrix.double()
    if x.size(0) < x.size(1):
        gram = x @ x.T
    else:
        gram = x.T @ x

    eigvals = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    singular = torch.sqrt(eigvals)
    total = singular.sum()
    if float(total) <= eps:
        return 0.0

    p = singular / total
    p = p[p > eps]
    entropy = -(p * torch.log(p)).sum()
    return float(torch.exp(entropy).item())

def collect_embeddings(args, model):
    frames = []
    utterance_sums = []
    seconds = 0.0
    max_seconds = float(args.max_seconds)
    rng = np.random.RandomState(int(args.seed))

    for url in args.subsets:
        dataset = torchaudio.datasets.LIBRISPEECH(str(args.librispeech_path), url=url)
        order = list(range(len(dataset)))
        rng.shuffle(order)
        for idx in tqdm(order, desc=f"embed/{url}"):
            if seconds >= max_seconds:
                break

            sample = dataset[idx]
            hidden = _extract_hidden(model, sample, int(args.layer))
            frames.append(hidden)
            utterance_sums.append(hidden.sum(dim=0))
            seconds += float(sample[0].numel()) / float(sample[1])

        if seconds >= max_seconds:
            break

    if not frames:
        raise RuntimeError("no embeddings collected; check --librispeech-path / --subsets")

    all_frames = torch.cat(frames, dim=0)
    utt_matrix = torch.stack(utterance_sums, dim=0)
    return all_frames, utt_matrix, seconds

def run_ger(args):
    model = _load_encoder(args)
    all_frames, utt_matrix, seconds = collect_embeddings(args, model)

    tqdm.write(
        f"collected {all_frames.shape[0]} frames x {all_frames.shape[1]} "
        f"from {utt_matrix.shape[0]} utterances ({seconds / 60.0:.1f} min audio)"
    )

    ger = effective_rank(all_frames)
    rankme_t = effective_rank(utt_matrix)

    report = {
        "checkpoint": str(args.checkpoint_path),
        "layer": int(args.layer),
        "model_size": args.model_size,
        "num_classes": str(args.num_classes),
        "label_rate": float(args.label_rate),
        "subsets": list(args.subsets),
        "seconds": float(seconds),
        "n_frames": int(all_frames.shape[0]),
        "n_utterances": int(utt_matrix.shape[0]),
        "dim": int(all_frames.shape[1]),
        "ger": ger,
        "rankme_t": rankme_t,
    }

    print(json.dumps(report, indent=2))
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

        tqdm.write(f"saved {args.out_json}")

def cli_main():
    parser = ArgumentParser(
        description=(
            "Compute Global Effective Rank (GER) and RankMe-t on HuBERT embeddings "
            "(Whetten et al., arXiv:2501.05966)."
        )
    )
    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        required=True,
        help="HuBERT pretrain Lightning checkpoint.",
    )
    parser.add_argument(
        "--librispeech-path",
        type=pathlib.Path,
        required=True,
        help="Path to LibriSpeech root.",
    )
    parser.add_argument(
        "--model-size",
        default="base",
        choices=["tiny", "base", "large", "xlarge"],
        help="Must match the checkpoint. (Default: base)",
    )
    parser.add_argument(
        "--num-classes",
        default="100",
        help="Codebook sizes used at pretrain. (Default: 100)",
    )
    parser.add_argument(
        "--label-rate",
        default=100.0,
        type=float,
        help="Label rate used to rebuild the encoder. (Default: 100)",
    )
    parser.add_argument(
        "--layer",
        default=12,
        type=int,
        help="1-based transformer layer. Paper uses 12 for ASR GER. (Default: 12)",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["train-clean-100"],
        help="Splits to sample audio from. (Default: train-clean-100)",
    )
    parser.add_argument(
        "--max-seconds",
        default=3600.0,
        type=float,
        help="Audio budget in seconds (~1h in the paper). (Default: 3600)",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Shuffle seed for utterance sampling. (Default: 0)",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Run feature extraction on CUDA.",
    )
    parser.add_argument(
        "--out-json",
        type=pathlib.Path,
        default=None,
        help="Optional path to save the JSON report.",
    )
    args = parser.parse_args()
    run_ger(args)

if __name__ == "__main__":
    cli_main()
