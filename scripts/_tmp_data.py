from types import SimpleNamespace

from data import ADDataset
import os

dataset_path = (
    "/home/usuaris/veussd/marc.casals/datasets/WAB_samples/"
)
parameters = SimpleNamespace(train_labels_path="/home/usuaris/veussd/marc.casals/datasets/WAB_samples/labels.csv",
                            validation_labels_path="/home/usuaris/veussd/marc.casals/datasets/WAB_samples/labels.csv",
                            audio_dir=os.path.join(dataset_path, "audios"),
                            window_secs=14,
                            stride_secs=7.0,
                            sample_rate=16000,
                            augmentation_prob=0.0,
                            num_folds=5,
                            text_feature_extractor="BERT_BASE_UNCASED",
                            dataset_transcriptions_dir=None,
                            padding_type="zero_pad",
                            target_classes=['exclude' 'bvFTD'],
                            random_seed=1234,
                            whisper_model_name="tiny",
                            whisper_language="es",
                            transcription_cache_dir=None,
                            )

data = ADDataset(
    input_parameters=parameters,


    # Patient-level split
    split="train",
    fold=1,

    # Classes to predict
    target_classes=["lvPPA", "nfPPA", "svPPA"],

    # These are already excluded by target_classes,
    # but may be stated explicitly for clarity.
    ignore_labels=["exclude", "bvFTD"],

)

print(data)
print("Label mapping:", data.label_map)
print("Number of segments:", len(data))
print("Segments per class:", data.class_counts)

# Load one 14-second segment
waveform, label, transcription_tokens = data[0]

print("Waveform shape:", waveform.shape)
print("Label:", label)
print("Label name:", data.label_names[label.item()])
print("Tokens shape:", transcription_tokens.shape)
