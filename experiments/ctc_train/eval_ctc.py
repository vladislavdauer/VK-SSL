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


def _transcript_from_sample(sample):
    if isinstance(sample, list) and len(sample) == 1:
        sample = sample[0]
    if isinstance(sample, (tuple, list)) and len(sample) > 2:
        return str(sample[2])
    raise TypeError(f"Unexpected sample format: {type(sample)}")


def run_eval(args):
    sp_model = spm.SentencePieceProcessor(model_file=str(args.sp_model_path))
    model = CTCTModule.load_from_checkpoint(args.checkpoint_path, sp_model=sp_model).eval()
    data_module = get_data_module(
        str(args.librispeech_path),
        str(args.global_stats_path),
        str(args.sp_model_path),
        sanity_check=bool(args.sanity_check),
    )

    if args.use_cuda:
        model = model.to(device="cuda")

    if args.sanity_check:
        model.train()
    else:
        model.eval()

    total_edit_distance = 0
    total_length = 0

    if args.sanity_check:
        dataloader = data_module.train_dataloader()
    else:
        dataloader = data_module.test_dataloader()

    with torch.no_grad():
        for idx, item in enumerate(dataloader):
            if args.sanity_check and idx >= 50:
                break

            if args.sanity_check:
                batch = item

                actual = []
                for tokens, length in zip(batch.targets, batch.target_lengths):
                    length = int(length.item())
                    target_ids = tokens[:length].detach().cpu().tolist()
                    actual.append(sp_model.decode(target_ids))
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
                print(f"WER  : {utt_wer:.4f}")

            current_wer = total_edit_distance / max(total_length, 1)
            logger.info(f"Processed elem {idx}; corpus WER: {current_wer:.4f}")

    final_wer = total_edit_distance / total_length if total_length > 0 else 0.0
    logger.info(f"Final corpus WER: {final_wer:.4f}")


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
        "--sanity_check",
        action="store_true",
        default=False,
        help="Run sanity check on 50 train batches.",
    )
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    cli_main()