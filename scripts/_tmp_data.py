from data import TrainDataset
from types import SimpleNamespace
import os

# Fake parameters object with only the attributes TrainDataset needs
parameters = SimpleNamespace(
    csv_path="/home/usuaris/veussd/roger.esteve.sanchez/WAB_samples/labels.csv",
    sample_rate=16000,
    text_feature_extractor="BERT_BASE_UNCASED",
    dataset_transcriptions_dir=None,
    padding_type="zero_pad",
    ignore_labels=None,
)
labels_lines = [
    "/path/to/audio.wav\t0",
]
dataset_path = "/home/usuaris/veussd/roger.esteve.sanchez/WAB_samples"
csv_path = os.path.join(dataset_path, 'labels.csv')

data = TrainDataset(
    labels_lines=labels_lines,
    input_parameters=parameters,
    random_crop_secs=2.0,
    augmentation_prob=0,
    waveforms_mean=None,
    waveforms_std=None,
    split="train",
    fold=1
)

print(data)
print(f"Number of files: {len(data)}")

# Try loading one sample
waveform, label, transcription_tokens = data[0]

print("Waveform shape:", waveform.shape)
print("Label:", label)
print("Tokens shape:", transcription_tokens.shape)