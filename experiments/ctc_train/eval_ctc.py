import logging
import pathlib
from argparse import ArgumentParser

import sentencepiece as spm

import torch
import torchaudio

from src.models.asr_lightning_module import CTCTModule
from src.data.librispeech_data_module import get_data_module

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())


def run_eval(args):
    sp_model = spm.SentencePieceProcessor(model_file=str(args.sp_model_path))
    model = CTCTModule.load_from_checkpoint(args.checkpoint_path, sp_model=sp_model).eval()
    data_module = get_data_module(
        str(args.librispeech_path),
        str(args.global_stats_path),
        str(args.sp_model_path)
    )

    if args.use_cuda:
        model = model.to(device="cuda")

    total_edit_distance = 0
    total_length = 0

    if args.subset == "val":
        dataloader = data_module.val_dataloader()
    else:
        dataloader = data_module.test_dataloader()

    with torch.no_grad():
        for idx, item in enumerate(dataloader):
            if args.subset == "val":
                batch = item
                if hasattr(batch, 'targets') and batch.targets is not None:
                    target_tokens = batch.targets.cpu().tolist()
                    actual = []
                    for tokens in target_tokens:
                        filtered = [t for t in tokens if t not in [0, 1, 2, 3]]
                        if filtered:
                            actual.append(sp_model.decode(filtered))
                        else:
                            actual.append("")
                    if len(actual) == 1:
                        actual = actual[0]
                else:
                    actual = ""
            else:
                batch, sample = item
                if isinstance(sample, (tuple, list)) and len(sample) > 2:
                    actual = sample[2]
                else:
                    actual = str(sample)

            predicted = model(batch)

            if isinstance(actual, list):
                for a, p in zip(actual, predicted if isinstance(predicted, list) else [predicted]):
                    total_edit_distance += compute_word_level_distance(p, a)
                    total_length += len(a.split()) if len(a.split()) > 0 else 1
            else:
                total_edit_distance += compute_word_level_distance(actual, predicted)
                total_length += len(actual.split()) if len(actual.split()) > 0 else 1

            if idx % 100 == 0:
                current_wer = total_edit_distance / total_length
                logger.info(f"Processed elem {idx}; Current WER: {current_wer:.4f}")

    final_wer = total_edit_distance / total_length
    logger.info(f"Final WER: {final_wer:.4f}")


def cli_main():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        help="Path to checkpoint to use for evaluation.",
        required=True,
    )
    parser.add_argument(
        "--global-stats-path",
        default=pathlib.Path("global_stats.json"),
        type=pathlib.Path,
        help="Path to JSON file containing feature means and stddevs.",
    )
    parser.add_argument(
        "--librispeech-path",
        type=pathlib.Path,
        help="Path to LibriSpeech datasets.",
        required=True,
    )
    parser.add_argument(
        "--sp-model-path",
        type=pathlib.Path,
        help="Path to SentencePiece model.",
        required=True,
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        default=False,
        help="Run using CUDA.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["test", "val"],
        default="test",
        help="Subset to evaluate on.",
    )
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    cli_main()