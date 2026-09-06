from collections import namedtuple
from typing import Dict, List, Sequence

import torch
import torchaudio

from src.data.data_transforms import _extract_labels

HubertBatch = namedtuple(
    "HubertBatch",
    ["inputs", "input_lengths", "targets", "target_lengths"],
)

HUBERT_LTR_VOCAB = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ'") + ["|"]
HUBERT_LTR_TO_ID = {char: idx for idx, char in enumerate(HUBERT_LTR_VOCAB)}
HUBERT_BLANK_ID = len(HUBERT_LTR_VOCAB)


def transcript_to_ltr(text: str) -> str:
    text = " ".join(str(text).upper().split())
    letters = []
    for char in text.replace(" ", "|"):
        if char in HUBERT_LTR_TO_ID:
            letters.append(char)
    if not letters:
        return "|"
    if letters[-1] != "|":
        letters.append("|")
    return " ".join(letters)


def encode_hubert_ltr(text: str) -> List[int]:
    tokens = transcript_to_ltr(text).split()
    ids = []
    for token in tokens:
        if token not in HUBERT_LTR_TO_ID:
            raise KeyError(f"Unknown HuBERT letter token {token!r} in transcript {text!r}")
        ids.append(HUBERT_LTR_TO_ID[token])
    return ids


def decode_hubert_ltr(token_ids: Sequence[int]) -> str:
    chars = []
    for idx in token_ids:
        idx = int(idx)
        if 0 <= idx < len(HUBERT_LTR_VOCAB):
            chars.append(HUBERT_LTR_VOCAB[idx])
    text = "".join(chars).replace("|", " ").strip()
    return " ".join(text.split())


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
    def __init__(
        self,
        label_maps: Sequence[Dict[str, List[int]]],
        sample_rate: int = 16000,
        label_rate: float = 100.0,
        max_sample_size: int = 250000,
        random_crop: bool = True,
    ):
        self.label_maps = list(label_maps)
        self.sample_rate = sample_rate
        self.label_rate = float(label_rate)
        self.max_sample_size = int(max_sample_size)
        self.random_crop = bool(random_crop)

    def _crop_wav_and_labels(self, wav: torch.Tensor, label_seqs: List[torch.Tensor]):
        n = int(wav.numel())
        if (not self.random_crop) or n <= self.max_sample_size:
            return wav, label_seqs

        start = int(torch.randint(0, n - self.max_sample_size + 1, (1,)).item())
        end = start + self.max_sample_size
        wav = wav[start:end]

        label_start = int(start * self.label_rate / self.sample_rate)
        label_end = int(end * self.label_rate / self.sample_rate)
        cropped = []
        for seq in label_seqs:
            cropped.append(seq[label_start:label_end].clone())
        return wav, cropped

    def __call__(self, samples: List):
        waves = []
        lengths = []
        codebook_seqs = [[] for _ in self.label_maps]

        for sample in samples:
            wav = waveform_16k(sample, self.sample_rate)
            utt_id = librispeech_utt_id(sample)
            label_seqs = []
            for label_map in self.label_maps:
                if utt_id not in label_map:
                    raise KeyError(f"Missing HuBERT labels for utterance {utt_id}")
                label_seqs.append(torch.tensor(label_map[utt_id], dtype=torch.long))

            wav, label_seqs = self._crop_wav_and_labels(wav, label_seqs)
            waves.append(wav)
            lengths.append(wav.numel())
            for k, seq in enumerate(label_seqs):
                codebook_seqs[k].append(seq)

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


class HubertCharFinetuneTransform:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def __call__(self, samples: List):
        waves = [waveform_16k(sample, self.sample_rate) for sample in samples]
        lengths = torch.tensor([wav.numel() for wav in waves], dtype=torch.long)
        inputs = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True)
        encoded = [encode_hubert_ltr(sample[2]) for sample in samples]
        target_lengths = torch.tensor([len(seq) for seq in encoded], dtype=torch.long)
        targets = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq, dtype=torch.long) for seq in encoded],
            batch_first=True,
            padding_value=HUBERT_BLANK_ID,
        )
        return HubertBatch(inputs, lengths, targets, target_lengths)


class HubertCharFinetuneTestTransform:
    def __init__(self, sample_rate: int = 16000):
        self.inner = HubertCharFinetuneTransform(sample_rate)

    def __call__(self, sample):
        return self.inner([sample]), [sample]


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
