import os
import pandas as pd
from transformers import AutoTokenizer, RobertaModel, Wav2Vec2Processor, Wav2Vec2Model, BertTokenizer, BertModel, DistilBertModel, AutoModel
import torch
import torchaudio
import unicodedata
import librosa
import numpy as np

from word_alignment import frame_bounds, token_audio_intervals

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Avaiable: bert, roberta, distilbert, stella, mistral, qwen
textual_model = 'distilbert'
audio_model = 'wav2vec2'
pauses = False

pauses_data = '_pauses' if pauses else ''
name_mapping_text = {
    'bert': '',
    'distilbert': 'distil',
    'roberta': 'roberta',
    'mistral': 'mistral',
    'qwen': 'qwen',
    'stella': 'stella'
}
textual_model_data = name_mapping_text.get(textual_model, '')
name_mapping_audio = {
    'wav2vec2': 'audio',
    'egemaps': 'egemaps',
    'mel': 'mel'
}
audio_model_data = '_' + name_mapping_audio.get(audio_model, '')

if textual_model == 'bert':
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
elif textual_model == 'roberta':
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base").to(device)
elif textual_model == 'distilbert':
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', use_fast=True)
    model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
elif textual_model == 'stella':
    tokenizer = AutoTokenizer.from_pretrained("NovaSearch/stella_en_1.5B_v5", trust_remote_code=True)
    model = AutoModel.from_pretrained("NovaSearch/stella_en_1.5B_v5", trust_remote_code=True)
elif textual_model == 'mistral':
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    # Need Access Token
    model = AutoModel.from_pretrained("mistralai/Mistral-7B-v0.1", use_auth_token=True)
elif textual_model == 'qwen':
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    # Need Access Token
    model = AutoModel.from_pretrained("Qwen/Qwen2.5-7B")

model.eval()

if audio_model == 'wav2vec2':
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    wav2vec_model.eval()
    segment_length = 50
elif audio_model == 'egemaps':
    import opensmile
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    segment_length = 10
else:
    segment_length = 50

dataset_path = '/home/usuaris/veussd/marc.casals/datasets/WAB_samples'
root_path = os.path.join(dataset_path, 'WAB_samples')
preprocessing_path = os.path.join(dataset_path, 'preprocessing')
word_level_dir = os.path.join(preprocessing_path, 'words')
embeddings_dir = os.path.join(preprocessing_path, 'embeddings')

textual_data = os.path.join(preprocessing_path, 'transcriptions.csv')
max_length = 200


def preprocess_text():

    os.makedirs(embeddings_dir, exist_ok=True)

    # Resolve transcript UIDs to the original audio filename, including its
    # extension. Whisper now processes WAV, MP3 and ALAC recordings.
    audio_paths = {}
    if audio_model != '':
        for filename in sorted(os.listdir(root_path)):
            audio_path = os.path.join(root_path, filename)
            if not filename.lower().endswith((".wav", ".mp3", ".alac")) or not os.path.isfile(audio_path):
                continue
            uid = os.path.splitext(filename)[0]
            if uid in audio_paths:
                raise ValueError(f"Multiple audio files share UID {uid!r} in {root_path}")
            audio_paths[uid] = audio_path

    # Read textual data from CSV
    df = pd.read_csv(textual_data, encoding='utf-8', dtype={'uid': str}, keep_default_na=False)


    row_data = 'transcription_pause' if pauses else 'transcription'

    df[row_data] = df[row_data].apply(lambda x: unicodedata.normalize("NFC", str(x)))

    completed_audios = 0

    # Transcript rows are identified by uid; diagnosis labels are not required.

    # Iteate over each row
    for index, row in df.iterrows():

        print(f"------------------------------------------")
        print(f"------------------------------------------")
        print(f"Processing {row['uid']}")


        # Get the transcription
        transcription = row[row_data]

        # Tokenize the transcription
        inputs_text = tokenizer(
            transcription,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        offsets = inputs_text.pop("offset_mapping")[0].tolist()
        inputs_text = inputs_text.to(device)

        # Get the embeddings
        with torch.no_grad():
            outputs_text = model(**inputs_text)

        # Save paired embeddings only after alignment succeeds.
        last_hidden_states_text = outputs_text.last_hidden_state.squeeze(0).cpu()

        if audio_model != '':
            audio_path = audio_paths.get(row['uid'])
            if audio_path is None:
                raise FileNotFoundError(f"No WAV, MP3 or ALAC recording for UID {row['uid']!r} in {root_path}")

            if audio_model == 'wav2vec2':
                wave_form, sample_rate = torchaudio.load(audio_path)

                # Convert stereo to mono if necessary
                if wave_form.shape[0] > 1:
                    wave_form = wave_form.mean(dim=0, keepdim=True)

                wave_form = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(wave_form)
                sample_rate = 16000
                wave_form = wave_form.squeeze(0)

                inputs_audio = processor(wave_form, sampling_rate=sample_rate, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs_audio = wav2vec_model(**inputs_audio)

                last_hidden_states_audio = outputs_audio.last_hidden_state.squeeze(0).cpu()
                processed_audio_tensor = torch.zeros((max_length, last_hidden_states_audio.shape[1]))

                if torch.isnan(last_hidden_states_audio).any():
                    last_hidden_states_audio = torch.nan_to_num(last_hidden_states_audio, nan=0.0)
            elif audio_model == 'egemaps':
                y, sr = librosa.load(audio_path)
                frame_size = 0.1

                frame_samples = int(frame_size * sr)  # Samples per frame
                frames = librosa.util.frame(y, frame_length=frame_samples, hop_length=frame_samples).T

                features = []
                for frame in frames:
                    features.append(smile.process_signal(frame, sr))

                features = np.vstack(features)

                features_audio = torch.tensor(features).float().to(device)
                print(f"Features shape: {features_audio.shape}")

                processed_audio_tensor = torch.zeros((max_length, features_audio.shape[1]))

                if torch.isnan(features_audio).any():
                    print(f"ERROR BEFORE in {row['uid']}: NaN values in features_audio")
                    features_audio = torch.nan_to_num(features_audio, nan=0.0)
            elif audio_model == 'mel':
                y, sr = librosa.load(audio_path)

                win_length = int(0.02 * sr)  # 20 ms en samples
                hop_length = int(0.02 * sr)  # 20 ms también para 50 segmentos por segundo
                n_mels = 80  # Número típico de filtros mel

                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=win_length, hop_length=hop_length, n_mels=n_mels)

                features_audio = torch.tensor(mel).float().permute(1,0)

                processed_audio_tensor = torch.zeros((max_length, features_audio.shape[1]))

                if torch.isnan(features_audio).any():
                    features_audio = torch.nan_to_num(features_audio, nan=0.0)

            features_audio = (last_hidden_states_audio if audio_model == 'wav2vec2' else features_audio).cpu()
            processed_audio_tensor[0] = features_audio.mean(dim=0)

            word_level_timestamp_path = os.path.join(word_level_dir, row['uid'] + '.csv')
            df_word_level = pd.read_csv(word_level_timestamp_path, dtype={'word': str}, keep_default_na=False)
            words = list(df_word_level[['word', 'start', 'end']].itertuples(index=False, name=None))
            intervals = token_audio_intervals(
                transcription, words, offsets,
                audio_duration=features_audio.shape[0] / segment_length,
            )

            n_audio_segments = 0
            for token_index, interval in enumerate(intervals):
                if interval is None:
                    continue
                first, last = frame_bounds(*interval, segment_length, features_audio.shape[0])
                processed_audio_tensor[token_index] = torch.clamp(
                    features_audio[first:last].mean(dim=0), min=-1e3, max=1e3,
                )
                n_audio_segments += 1

            expected_segments = sum(start != end for start, end in offsets)
            print(f"Aligned audio tokens: {n_audio_segments}/{expected_segments}")
            print(f"Total tokens including special tokens: {int(inputs_text['attention_mask'].sum())}")
            if n_audio_segments != expected_segments or not torch.isfinite(processed_audio_tensor).all():
                raise ValueError(f"Invalid aligned audio embeddings for {row['uid']}")

            torch.save(processed_audio_tensor, os.path.join(embeddings_dir, row['uid'] + textual_model_data + pauses_data + audio_model_data + '.pt'))


        torch.save(last_hidden_states_text, os.path.join(embeddings_dir, row['uid'] + textual_model_data + pauses_data + '.pt'))

        completed_audios += 1

        print(f"------------------------------------------")
        print(f"CORRECTLY PROCESSED RECORDING")
        print(f"Completed audios: {completed_audios}")

if __name__ == "__main__":
    raise SystemExit(preprocess_text())
