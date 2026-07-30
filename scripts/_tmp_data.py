from types import SimpleNamespace

from data import ADDataset


dataset_path = (
    "/home/usuaris/veussd/roger.esteve.sanchez/WAB_samples"
)
parameters = SimpleNamespace(csv_path=f"{dataset_path}/labels.csv",
                            audio_dir=dataset_path,
                            window_secs=14,
                            stride_secs=7.0,
                            sample_rate=16000,
                            augmentation_prob=0.0,
                            whisper_model_name="tiny",
                            whisper_language="es",
                            num_folds=5,
                            text_feature_extractor="BERT_BASE_UNCASED",
                            dataset_transcriptions_dir=None,
                            transcription_cache_dir=None,
                            padding_type="zero_pad",
                            target_classes=['exclude', 'bvFTD'],
                            random_seed=1234 )

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


    # None automatically selects:
    # WAB_samples/trans_w14_s7.00

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
