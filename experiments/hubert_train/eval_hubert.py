import logging
import pathlib
from argparse import ArgumentParser

import sentencepiece as spm

import torch
import torchaudio
from torch.utils.data import DataLoader

from src.data.hubert_data_module import get_hubert_finetune_data_module
from src.data.hubert_transforms import decode_hubert_ltr
from src.data.librispeech_data_module import TransformDataset
from src.models.hubert_lightning_module import HubertCTCModule

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger()


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())


def _transcript_from_sample(sample):
    if isinstance(sample, list) and len(sample) == 1:
        sample = sample[0]
    if isinstance(sample, (tuple, list)) and len(sample) > 2:
        return str(sample[2])
    raise TypeError(f"Unexpected sample format: {type(sample)}")


def _test_dataloader(data_module, url):
    dataset = data_module.librispeech_cls(data_module.librispeech_path, url=url)
    dataset = TransformDataset(dataset, data_module.test_transform)
    return DataLoader(dataset, batch_size=None)


def _decode_targets(batch, label_type, sp_model):
    texts = []
    for tokens, length in zip(batch.targets, batch.target_lengths):
        length = int(length.item())
        target_ids = tokens[:length].detach().cpu().tolist()
        if label_type == "spm":
            texts.append(sp_model.decode(target_ids))
        else:
            texts.append(decode_hubert_ltr(target_ids))
    return texts


def eval_dataloader(model, label_type, sp_model, dataloader, subset_name, sanity_check=False):
    total_edit_distance = 0
    total_length = 0

    with torch.no_grad():
        for idx, item in enumerate(dataloader):
            if sanity_check and idx >= 50:
                break

            if sanity_check:
                batch = item
                actual = _decode_targets(batch, label_type, sp_model)
            else:
                batch, sample = item
                actual = [_transcript_from_sample(sample)]

            predicted = model(batch)
            if isinstance(predicted, str):
                predicted = [predicted]

            for a, p in zip(actual, predicted):
                utt_edits = compute_word_level_distance(p, a)
                utt_words = max(len(a.split()), 1)
                utt_wer = utt_edits / utt_words

                total_edit_distance += utt_edits
                total_length += utt_words

                print(f"ACTUAL   : {a}")
                print(f"PREDICTED: {p}")
                print(f"WER      : {utt_wer:.4f}\n")

            current_wer = total_edit_distance / max(total_length, 1)
            logger.info(f"[{subset_name}] elem {idx}; corpus WER: {current_wer:.4f}")

    final_wer = total_edit_distance / total_length if total_length > 0 else 0.0
    logger.info(f"[{subset_name}] Final corpus WER: {final_wer:.4f}")
    return final_wer


def run_eval(args):
    label_type = str(getattr(args, "label_type", "char")).lower()
    sp_model = None
    if label_type == "spm":
        sp_model = spm.SentencePieceProcessor(model_file=str(args.sp_model_path))

    model = HubertCTCModule.load_from_checkpoint(
        args.checkpoint_path,
        sp_model=sp_model,
        strict=False,
    ).eval()
    data_module = get_hubert_finetune_data_module(
        str(args.librispeech_path),
        sp_model_path=str(args.sp_model_path) if args.sp_model_path else None,
        label_type=label_type,
        sanity_check=bool(args.sanity_check),
    )

    if args.use_cuda:
        model = model.to(device="cuda")

    if args.sanity_check:
        model.train()
        eval_dataloader(
            model,
            label_type,
            sp_model,
            data_module.train_dataloader(),
            "train-sanity",
            sanity_check=True,
        )
        return

    model.eval()
    results = {}
    for url in args.subsets:
        loader = _test_dataloader(data_module, url)
        results[url] = eval_dataloader(model, label_type, sp_model, loader, url)

    for url, wer in results.items():
        logger.info(f"{url}: {wer:.4f}")


def cli_main():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        help="Path to checkpoint to use for evaluation.",
        required=True,
    )
    parser.add_argument(
        "--librispeech-path",
        type=pathlib.Path,
        help="Path to LibriSpeech datasets.",
        required=True,
    )
    parser.add_argument(
        "--sp-model-path",
        default=None,
        type=pathlib.Path,
        help="Path to SentencePiece model (required for --label-type spm).",
    )
    parser.add_argument(
        "--label-type",
        default="char",
        choices=["char", "spm"],
        help="CTC label type. (Default: char)",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        default=False,
        help="Run using CUDA.",
    )
    parser.add_argument(
        "--sanity_check",
        action="store_true",
        default=False,
        help="Run sanity check on 50 train batches.",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["test-clean", "test-other"],
        choices=["test-clean", "test-other"],
        help="LibriSpeech test subsets to evaluate. (Default: test-clean test-other)",
    )
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    cli_main()
