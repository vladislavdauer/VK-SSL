from collections import namedtuple
from typing import Dict, List, Sequence

import torch
import torchaudio

from src.data.data_transforms import _extract_labels

HubertBatch = namedtuple(
    "HubertBatch",
    ["inputs", "input_lengths", "targets", "target_lengths"],
)


def librispeech_utt_id(sample) -> str:
    speaker_id = sample[3]
    chapter_id = sample[4]
    utterance_id = sample[5]
    return f"{speaker_id}-{chapter_id}-{int(utterance_id):04d}"


def waveform_16k(sample, sample_rate: int = 16000) -> torch.Tensor:
    wav = sample[0].squeeze(0).float()
    src_sr = int(sample[1])
    if src_sr != sample_rate:
        wav = torchaudio.functional.resample(wav, src_sr, sample_rate)

    return wav


def load_km_file(path: str) -> Dict[str, List[int]]:
    labels = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            utt_id = parts[0]
            labels[utt_id] = [int(v) for v in parts[1:]]

    return labels


class HubertPretrainTransform:
    def __init__(self, label_maps: Sequence[Dict[str, List[int]]], sample_rate: int = 16000):
        self.label_maps = list(label_maps)
        self.sample_rate = sample_rate

    def __call__(self, samples: List):
        waves = []
        lengths = []
        codebook_seqs = [[] for _ in self.label_maps]

        for sample in samples:
            wav = waveform_16k(sample, self.sample_rate)
            waves.append(wav)
            lengths.append(wav.numel())
            utt_id = librispeech_utt_id(sample)
            for k, label_map in enumerate(self.label_maps):
                if utt_id not in label_map:
                    raise KeyError(f"Missing HuBERT labels for utterance {utt_id}")

                codebook_seqs[k].append(torch.tensor(label_map[utt_id], dtype=torch.long))

        inputs = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True)
        input_lengths = torch.tensor(lengths, dtype=torch.long)
        targets = [
            torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=-1)
            for seqs in codebook_seqs
        ]
        target_lengths = [
            torch.tensor([seq.numel() for seq in seqs], dtype=torch.long)
            for seqs in codebook_seqs
        ]
        return HubertBatch(inputs, input_lengths, targets, target_lengths)


class DummyHubertPretrainTransform:
    def __init__(self, num_classes: Sequence[int], label_rate: float = 100.0, sample_rate: int = 16000):
        self.num_classes = list(num_classes)
        self.label_rate = label_rate
        self.sample_rate = sample_rate

    def __call__(self, samples: List):
        waves = []
        lengths = []
        codebook_seqs = [[] for _ in self.num_classes]
        hop = max(int(self.sample_rate / self.label_rate), 1)

        for sample in samples:
            wav = waveform_16k(sample, self.sample_rate)
            waves.append(wav)
            lengths.append(wav.numel())
            n_frames = max(int(wav.numel() / hop), 1)
            for k, n_class in enumerate(self.num_classes):
                codebook_seqs[k].append(torch.randint(0, n_class, (n_frames,), dtype=torch.long))

        inputs = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True)
        input_lengths = torch.tensor(lengths, dtype=torch.long)

        targets = [
            torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=-1)
            for seqs in codebook_seqs
        ]

        target_lengths = [
            torch.tensor([seq.numel() for seq in seqs], dtype=torch.long)
            for seqs in codebook_seqs
        ]

        return HubertBatch(inputs, input_lengths, targets, target_lengths)


class HubertFinetuneTransform:
    def __init__(self, sp_model_path: str, sample_rate: int = 16000):
        import sentencepiece as spm

        self.sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.sample_rate = sample_rate

    def __call__(self, samples: List):
        waves = [waveform_16k(sample, self.sample_rate) for sample in samples]
        lengths = torch.tensor([wav.numel() for wav in waves], dtype=torch.long)
        inputs = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True)
        targets, target_lengths = _extract_labels(self.sp_model, samples)

        return HubertBatch(inputs, lengths, targets, target_lengths)


class HubertFinetuneTestTransform:
    def __init__(self, sp_model_path: str, sample_rate: int = 16000):
        self.inner = HubertFinetuneTransform(sp_model_path, sample_rate)

    def __call__(self, sample):
        return self.inner([sample]), [sample]
