import json
import pathlib
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from src.data.duration_cache import open_subset
from src.data.hubert_transforms import (
    HUBERT_LTR_TO_ID,
    HUBERT_LTR_VOCAB,
    encode_hubert_ltr,
    librispeech_utt_id,
    load_km_file,
    waveform_16k,
)
from src.models.hubert.config import HUBERT_SIZES, get_hubert_config
from src.models.hubert.hubert_model import HubertModel, load_encoder_state
from src.models.hubert.kmeans import extract_mfcc_39

MFCC_RATE = 100.0

def _fileids(dataset):
    if isinstance(dataset, torchaudio.datasets.LibriLightLimited):
        return [fileid for _, fileid in dataset._fileids_paths]

    return list(dataset._walker)

def select_utterances(args):
    rng = np.random.RandomState(int(args.seed))
    datasets = []
    by_speaker = defaultdict(list)
    for url in args.subsets:
        dataset = open_subset(args.librispeech_path, url)
        datasets.append(dataset)
        for idx, fileid in enumerate(_fileids(dataset)):
            by_speaker[fileid.split("-")[0]].append((len(datasets) - 1, idx))

    speakers = sorted(by_speaker)
    if not speakers:
        raise RuntimeError("no utterances found; check --librispeech-path / --subsets")

    if int(args.num_speakers) < len(speakers):
        speakers = sorted(rng.choice(speakers, int(args.num_speakers), replace=False).tolist())

    picks = []
    for speaker in speakers:
        items = by_speaker[speaker]
        order = rng.permutation(len(items))[: int(args.utts_per_speaker)]
        picks.extend(items[i] for i in sorted(order))

    return datasets, picks

def _checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    hparams = dict(ckpt.get("hyper_parameters", {}) or {}) if isinstance(ckpt, dict) else {}
    return state, hparams

def load_model(args):
    state, hparams = ({}, {})
    if args.checkpoint_path is not None:
        state, hparams = _checkpoint(args.checkpoint_path)

    for key in ("model_size", "num_classes", "label_rate"):
        if getattr(args, key) is None:
            setattr(args, key, hparams.get(key, {"model_size": "small", "num_classes": "100", "label_rate": 100.0}[key]))

    num_classes = [int(v) for v in str(args.num_classes).split(",") if str(v).strip()]
    cfg = get_hubert_config(args.model_size, num_classes=num_classes, label_rate=float(args.label_rate))
    torch.manual_seed(int(args.seed))
    model = HubertModel(cfg)
    info = {"init": "random"}
    ctc_out = None
    if args.checkpoint_path is not None:
        report = load_encoder_state(model, args.checkpoint_path)
        info = {
            "init": str(args.checkpoint_path),
            "loaded_tensors": f"{report['loaded']}/{report['expected']}",
            "checkpoint_step": report["global_step"],
            "checkpoint_max_steps": hparams.get("max_steps"),
            "finetuned": "ctc_out.weight" in state,
        }
        if "ctc_out.weight" in state:
            weight = state["ctc_out.weight"]
            if weight.size(0) == len(HUBERT_LTR_VOCAB) + 1:
                ctc_out = torch.nn.Linear(weight.size(1), weight.size(0))
                ctc_out.load_state_dict({"weight": weight, "bias": state["ctc_out.bias"]})
                ctc_out.eval()
            else:
                tqdm.write(f"CTC head has {weight.size(0)} outputs (not char), letter colouring disabled")

    if args.layer is None:
        args.layer = cfg.encoder_layers
    if not 1 <= int(args.layer) <= cfg.encoder_layers:
        raise ValueError(f"--layer must be in [1, {cfg.encoder_layers}], got {args.layer}")

    model.eval()
    if args.use_cuda and torch.cuda.is_available():
        model = model.cuda()
        ctc_out = ctc_out.cuda() if ctc_out is not None else None

    return model, ctc_out, info

def _resample_labels(codes, n_frames, src_rate, frame_rate):
    if len(codes) == 0:
        return np.full(n_frames, -1, dtype=np.int64)

    idx = np.minimum((np.arange(n_frames) * src_rate / frame_rate).astype(np.int64), len(codes) - 1)
    return np.asarray(codes, dtype=np.int64)[idx]

def _letters(ctc_out, last_hidden, transcript, source):
    log_probs = torch.log_softmax(ctc_out(last_hidden).float(), dim=-1).cpu()
    blank = log_probs.size(-1) - 1
    if source == "greedy":
        path = log_probs.argmax(dim=-1)
    else:
        targets = torch.tensor([encode_hubert_ltr(transcript)], dtype=torch.int32)
        try:
            aligned, _ = torchaudio.functional.forced_align(log_probs.unsqueeze(0), targets, blank=blank)
        except RuntimeError:
            return None
        path = aligned[0]

    path = path.numpy().astype(np.int64)
    path[(path == blank) | (path == HUBERT_LTR_TO_ID["|"])] = -1
    return path

def extract(args, model, ctc_out, datasets, picks):
    rng = np.random.RandomState(int(args.seed))
    km = None
    if args.km_path is not None:
        import joblib

        km = joblib.load(args.km_path)
    km_labels = load_km_file(str(args.km_labels)) if args.km_labels is not None else None
    phones = load_km_file(str(args.phone_alignments)) if args.phone_alignments else None

    device = next(model.parameters()).device
    downsample = model.feature_extractor.downsampling_factor
    frame_rate = float(model.cfg.sample_rate) / downsample
    n_layers = model.cfg.encoder_layers
    out = defaultdict(list)
    utt_means, utt_speakers, utt_ids = [], [], []
    stats = {"letters_failed": 0, "phones_mismatched": 0, "phones_missing": 0, "km_labels_missing": 0}

    for u, (d, idx) in enumerate(tqdm(picks, desc="embed")):
        sample = datasets[d][idx]
        utt_id = librispeech_utt_id(sample)
        wav = waveform_16k(sample, model.cfg.sample_rate)
        source = wav.unsqueeze(0).to(device)
        lengths = torch.tensor([wav.numel()], dtype=torch.long, device=device)
        with torch.no_grad():
            hidden, feat_lengths, _ = model.extract_features(source, lengths, tgt_layer=int(args.layer) - 1)
            t = int(feat_lengths[0].item())
            frames = hidden[0, :t]
            letters = None
            if ctc_out is not None:
                last = hidden if int(args.layer) == n_layers else model.extract_features(source, lengths)[0]
                letters = _letters(ctc_out, last[0, :t], sample[2], args.letter_source)
                if letters is None:
                    stats["letters_failed"] += 1

        frames = frames.float().cpu().numpy()
        utt_means.append(frames.mean(axis=0))
        utt_speakers.append(utt_id.split("-")[0])
        utt_ids.append(utt_id)

        keep = np.sort(rng.choice(t, size=min(t, int(args.frames_per_utt)), replace=False))
        out["frames"].append(frames[keep])
        out["utt"].append(np.full(len(keep), u, dtype=np.int64))

        if km is not None:
            mfcc = extract_mfcc_39(wav, model.cfg.sample_rate).numpy()
            codes = km.predict(mfcc.astype(km.cluster_centers_.dtype))
            out["teacher"].append(_resample_labels(codes, t, MFCC_RATE, frame_rate)[keep])
        elif km_labels is not None:
            codes = km_labels.get(utt_id)
            if codes is None:
                stats["km_labels_missing"] += 1
                codes = []
            out["teacher"].append(_resample_labels(codes, t, float(args.km_label_rate), frame_rate)[keep])

        if phones is not None:
            codes = phones.get(utt_id)
            seconds = wav.numel() / float(model.cfg.sample_rate)
            if codes is None:
                stats["phones_missing"] += 1
                codes = []
            elif abs(len(codes) / float(args.phone_rate) - seconds) > 0.25:
                stats["phones_mismatched"] += 1
                codes = []
            out["phone"].append(_resample_labels(codes, t, float(args.phone_rate), frame_rate)[keep])

        if ctc_out is not None:
            out["letter"].append(letters[keep] if letters is not None else np.full(len(keep), -1, dtype=np.int64))

    data = {key: np.concatenate(value) for key, value in out.items()}
    speaker_names = sorted(set(utt_speakers))
    speaker_ids = np.array([speaker_names.index(s) for s in utt_speakers], dtype=np.int64)
    data["speaker"] = speaker_ids[data["utt"]]
    data["utt_means"] = np.stack(utt_means)
    data["utt_speaker"] = speaker_ids
    return data, speaker_names, utt_ids, stats

def knn_purity(features, labels, groups, k=10, metric="cosine"):
    from sklearn.neighbors import NearestNeighbors

    valid = labels >= 0
    features, labels, groups = features[valid], labels[valid], groups[valid]
    if len(labels) <= k or len(np.unique(labels)) < 2:
        return None

    n_query = min(len(labels), k + int(np.bincount(groups).max()))
    _, idx = NearestNeighbors(n_neighbors=n_query, metric=metric).fit(features).kneighbors(features)
    hits = []
    for i in range(len(labels)):
        others = idx[i][groups[idx[i]] != groups[i]][:k]
        if len(others):
            hits.append(float(np.mean(labels[others] == labels[i])))

    p = np.bincount(labels) / float(len(labels))
    return {"knn_purity": float(np.mean(hits)), "chance": float((p**2).sum()), "k": k, "n": int(len(labels))}

def _colors(n):
    import matplotlib

    if n <= 10:
        return [matplotlib.colormaps["tab10"](i) for i in range(n)]
    if n <= 20:
        return [matplotlib.colormaps["tab20"](i) for i in range(n)]

    cmap = matplotlib.colormaps["turbo"]
    order = np.random.RandomState(0).permutation(n)
    return [cmap(0.05 + 0.9 * order[i] / float(n - 1)) for i in range(n)]

def scatter(ax, coords, labels, names, title, point_size, annotate=False, max_legend=30):
    valid = labels >= 0
    if (~valid).any():
        ax.scatter(coords[~valid, 0], coords[~valid, 1], s=point_size, c="lightgrey", alpha=0.3, linewidths=0, rasterized=True)

    classes = np.unique(labels[valid])
    for color, c in zip(_colors(len(classes)), classes):
        pts = coords[labels == c]
        ax.scatter(pts[:, 0], pts[:, 1], s=point_size, color=color, alpha=0.7, linewidths=0, label=names[c], rasterized=True)
        if annotate:
            cx, cy = np.median(pts, axis=0)
            ax.text(cx, cy, names[c], fontsize=8, weight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))

    if len(classes) <= max_legend and not annotate:
        ax.legend(markerscale=max(1.0, 12.0 / max(point_size, 1.0)), fontsize=6, ncol=2, frameon=False, loc="best")

    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

def _purity_title(name, purity):
    if purity is None:
        return name
    return f"{name}\nkNN@{purity['k']} purity {purity['knn_purity']:.3f} (chance {purity['chance']:.3f})"

def run_umap(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap
    from sklearn.cluster import MiniBatchKMeans

    model, ctc_out, info = load_model(args)
    datasets, picks = select_utterances(args)
    data, speaker_names, utt_ids, stats = extract(args, model, ctc_out, datasets, picks)
    frames = data["frames"]

    normed = frames / np.linalg.norm(frames, axis=1, keepdims=True).clip(min=1e-8)
    n_clusters = min(int(args.n_clusters), len(frames))
    data["emb_kmeans"] = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=int(args.seed), n_init=3, batch_size=4096
    ).fit_predict(normed).astype(np.int64)

    tqdm.write(f"UMAP on {frames.shape[0]} frames x {frames.shape[1]} dims, {len(utt_ids)} utterances")
    reducer_kwargs = dict(
        n_neighbors=int(args.n_neighbors), min_dist=float(args.min_dist), metric=args.metric,
        random_state=int(args.seed),
    )
    coords = umap.UMAP(**reducer_kwargs).fit_transform(frames)
    utt_neighbors = max(2, min(int(args.n_neighbors), len(utt_ids) - 1))
    utt_coords = umap.UMAP(**{**reducer_kwargs, "n_neighbors": utt_neighbors}).fit_transform(data["utt_means"])

    letter_names = list(HUBERT_LTR_VOCAB)
    views = [("speaker", "frames by speaker", speaker_names, False)]
    if "letter" in data:
        views.append(("letter", f"frames by reference letter (CTC {args.letter_source})", letter_names, True))
    if "phone" in data:
        mapping = {}
        if args.phone_mapping is not None:
            with open(args.phone_mapping, encoding="utf-8") as handle:
                mapping = {int(v): k for k, v in json.load(handle).items()}
        n_phone = int(data["phone"].max()) + 1 if (data["phone"] >= 0).any() else 0
        views.append(("phone", "frames by phone (alignment)", [mapping.get(i, str(i)) for i in range(n_phone)], True))
    if "teacher" in data:
        n_teacher = int(data["teacher"].max()) + 1 if (data["teacher"] >= 0).any() else 0
        views.append(("teacher", "frames by teacher k-means cluster", [str(i) for i in range(n_teacher)], False))
    views.append(("emb_kmeans", f"frames by embedding k-means (k={n_clusters})", [str(i) for i in range(n_clusters)], False))

    purity = {}
    for key, *_ in views:
        if (data[key] >= 0).any():
            purity[key] = knn_purity(frames, data[key], data["utt"], k=int(args.knn), metric=args.metric)
    purity["utterance_speaker"] = knn_purity(
        data["utt_means"], data["utt_speaker"], np.arange(len(utt_ids)), k=min(int(args.knn), 5), metric=args.metric
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model_size} {model.cfg.encoder_embed_dim}-d, layer {args.layer}/{model.cfg.encoder_layers}, init: {args.label or info['init']}"
    panels = []
    for key, title, names, annotate in views:
        if not (data[key] >= 0).any():
            continue
        fig, ax = plt.subplots(figsize=(7, 6))
        scatter(ax, coords, data[key], names, _purity_title(title, purity.get(key)), float(args.point_size), annotate=annotate)
        fig.suptitle(tag, fontsize=9)
        fig.tight_layout()
        fig.savefig(args.out_dir / f"umap_frames_{key}.png", dpi=150)
        plt.close(fig)
        panels.append((coords, data[key], names, title, key, annotate))

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter(ax, utt_coords, data["utt_speaker"], speaker_names,
            _purity_title("utterances (mean-pooled) by speaker", purity["utterance_speaker"]), 18.0)
    fig.suptitle(tag, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_dir / "umap_utterances_speaker.png", dpi=150)
    plt.close(fig)
    panels.append((utt_coords, data["utt_speaker"], speaker_names, "utterances (mean-pooled) by speaker", "utterance_speaker", False))

    ncols = min(3, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows), squeeze=False)
    for ax, (xy, labels, names, title, key, annotate) in zip(axes.flat, panels):
        size = 18.0 if key == "utterance_speaker" else float(args.point_size)
        scatter(ax, xy, labels, names, _purity_title(title, purity.get(key)), size, annotate=annotate, max_legend=0)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle(tag, fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out_dir / "umap_overview.png", dpi=130)
    plt.close(fig)

    arrays = {k: v for k, v in data.items() if k not in ("frames", "utt_means")}
    if args.save_embeddings:
        arrays["frames"] = frames
        arrays["utt_means"] = data["utt_means"]
    np.savez_compressed(
        args.out_dir / "umap_points.npz", coords=coords, utt_coords=utt_coords,
        utt_ids=np.array(utt_ids), speaker_names=np.array(speaker_names), **arrays,
    )
    meta = {
        **info,
        "label": args.label,
        "model_size": args.model_size,
        "embed_dim": int(model.cfg.encoder_embed_dim),
        "layer": int(args.layer),
        "num_layers": int(model.cfg.encoder_layers),
        "subsets": list(args.subsets),
        "n_speakers": len(speaker_names),
        "n_utterances": len(utt_ids),
        "n_frames": int(frames.shape[0]),
        "umap": {"n_neighbors": int(args.n_neighbors), "min_dist": float(args.min_dist), "metric": args.metric},
        "letter_source": args.letter_source if "letter" in data else None,
        "purity": purity,
        "stats": stats,
        "seed": int(args.seed),
    }
    with open(args.out_dir / "umap_meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    tqdm.write(json.dumps({"purity": purity, "stats": stats}, indent=2))
    tqdm.write(f"saved plots to {args.out_dir}")
    if stats["phones_mismatched"] and stats["phones_mismatched"] >= len(utt_ids) // 2:
        tqdm.write(
            "WARNING: most phone label lengths do not match the audio at --phone-rate; "
            "librispeech_finetuning/phones/*.txt are phone sequences, not frame alignments"
        )
    return meta

def cli_main():
    parser = ArgumentParser(
        description="UMAP of HuBERT frame / utterance embeddings, coloured by speaker, cluster, letter or phone."
    )
    init = parser.add_mutually_exclusive_group(required=True)
    init.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        help="HuBERT pre-train or fine-tune (HubertCTCModule) Lightning checkpoint.",
    )
    init.add_argument(
        "--random-init",
        action="store_true",
        help="Untrained encoder of --model-size, as a reference for the learned ones.",
    )
    parser.add_argument("--librispeech-path", type=pathlib.Path, required=True, help="LibriSpeech / Libri-Light root.")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True, help="Directory for PNGs, umap_points.npz, umap_meta.json.")
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["dev-clean"],
        help="Splits to sample from, e.g. dev-clean or ll-10h. (Default: dev-clean)",
    )
    parser.add_argument(
        "--model-size",
        default=None,
        choices=HUBERT_SIZES,
        help="Default: read from checkpoint hparams, else small.",
    )
    parser.add_argument("--num-classes", default=None, help="Default: read from checkpoint hparams, else 100.")
    parser.add_argument("--label-rate", default=None, type=float, help="Default: read from checkpoint hparams, else 100.")
    parser.add_argument("--layer", default=None, type=int, help="1-based transformer layer. (Default: last layer)")
    parser.add_argument("--num-speakers", default=20, type=int, help="Speakers sampled. (Default: 20)")
    parser.add_argument("--utts-per-speaker", default=10, type=int, help="Utterances per speaker. (Default: 10)")
    parser.add_argument("--frames-per-utt", default=100, type=int, help="Random frames kept per utterance. (Default: 100)")
    parser.add_argument(
        "--km-path",
        type=pathlib.Path,
        default=None,
        help="MFCC k-means model (learn_kmeans.py km_100.bin) to colour frames by iteration-1 teacher cluster.",
    )
    parser.add_argument(
        "--km-labels",
        type=pathlib.Path,
        default=None,
        help="Dumped teacher labels (dump_labels.py all.km), used when --km-path is not given.",
    )
    parser.add_argument("--km-label-rate", default=100.0, type=float, help="Rate of --km-labels in Hz. (Default: 100)")
    parser.add_argument(
        "--phone-alignments",
        type=pathlib.Path,
        default=None,
        help="Frame-level phone alignments, one line per utterance: '<utt-id> id id ...' at --phone-rate.",
    )
    parser.add_argument("--phone-rate", default=100.0, type=float, help="Frame rate of --phone-alignments. (Default: 100)")
    parser.add_argument("--phone-mapping", type=pathlib.Path, default=None, help="JSON {phone: id} for legend names.")
    parser.add_argument(
        "--letter-source",
        default="align",
        choices=["align", "greedy"],
        help="Fine-tuned checkpoints: 'align' = CTC forced alignment of the reference transcript, "
        "'greedy' = argmax predictions. (Default: align)",
    )
    parser.add_argument("--n-clusters", default=50, type=int, help="k-means on the embeddings. (Default: 50)")
    parser.add_argument("--n-neighbors", default=30, type=int, help="UMAP n_neighbors. (Default: 30)")
    parser.add_argument("--min-dist", default=0.1, type=float, help="UMAP min_dist. (Default: 0.1)")
    parser.add_argument("--metric", default="cosine", help="UMAP / kNN metric. (Default: cosine)")
    parser.add_argument("--knn", default=10, type=int, help="k for kNN label purity. (Default: 10)")
    parser.add_argument("--point-size", default=2.0, type=float, help="Scatter point size. (Default: 2)")
    parser.add_argument("--label", default=None, help="Run name shown in plot titles.")
    parser.add_argument("--save-embeddings", action="store_true", help="Also store raw frames in umap_points.npz.")
    parser.add_argument("--seed", default=0, type=int, help="Sampling / UMAP seed. (Default: 0)")
    parser.add_argument("--use-cuda", action="store_true", help="Run the encoder on CUDA.")
    args = parser.parse_args()
    run_umap(args)

if __name__ == "__main__":
    cli_main()
