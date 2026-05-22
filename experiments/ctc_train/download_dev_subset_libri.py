import os
import torchaudio


os.makedirs("./librispeech", exist_ok=True)

dataset = torchaudio.datasets.LIBRISPEECH(
    root="./librispeech",
    url="dev-clean",
    folder_in_archive="LibriSpeech",
    download=True
)
print("DataSet size", len(dataset))
waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = dataset[0]
print(f"Waveform shape: {waveform.shape} | Transcript: {transcript[:50]}...")

output_path = "./librispeech/sample_0.wav"
torchaudio.save(output_path, waveform, sample_rate)