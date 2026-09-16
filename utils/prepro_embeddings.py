import os
import pandas as pd
from transformers import AutoTokenizer, RobertaModel, Wav2Vec2Processor, Wav2Vec2Model, BertTokenizer, BertModel, DistilBertModel, AutoModel
import torch
import torchaudio
import opensmile
import unicodedata
import librosa
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Avaiable: bert, roberta, distilbert, stella, mistral, qwen
textual_model = ''
audio_model = ''
pauses = False

pauses_data = '_pauses' if pauses else ''
name_mapping_text = {
    'bert': '',
    'distil': 'distil',
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
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
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
    segment_length = 50
elif audio_model == 'egemaps':
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    segment_length = 10
else:
    segment_length = 50

dataset_path = '/home/usuaris/veussd/marc.casals/datasets/WAB_samples'
root_path = os.path.join(dataset_path, 'audios')
preprocessing_path = os.path.join(dataset_path, 'preprocessing')
word_level_dir = os.path.join(preprocessing_path, 'words')
embeddings_dir = os.path.join(preprocessing_path, 'embeddings')

textual_data = os.path.join(preprocessing_path, 'transcriptions.csv')
max_length = 200


def preprocess_text():

    os.makedirs(embeddings_dir, exist_ok=True)

    # Read textual data from CSV
    df = pd.read_csv(textual_data, encoding='utf-8', dtype={'uid': str})

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
            max_length=max_length
        ).to(device)

        # Get the embeddings
        with torch.no_grad():
            outputs_text = model(**inputs_text)

        # Save the embeddings
        last_hidden_states_text = outputs_text.last_hidden_state.squeeze(0).cpu()
        torch.save(last_hidden_states_text, os.path.join(embeddings_dir, row['uid'] + textual_model_data + pauses_data + '.pt'))

        if audio_model != '':
            audio_path = os.path.join(root_path, row['uid'] + '.wav')

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

            processed_audio_tensor[0] = features_audio.mean(dim=0)

            # Tokenize and prepare inputs
            inputs_offset = tokenizer(
                transcription,
                return_tensors="pt",
                return_offsets_mapping=True,  # Get token-to-offset mappings
                padding="max_length",
                truncation=True,
                max_length=max_length
            ).to(device)


            # Extract word-to-token mapping
                # print(text)
            input_ids = inputs_offset["input_ids"][-1]
            offset_mapping = inputs_offset["offset_mapping"][-1]

            tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
            word_mapping = []

            current_word = ""
            current_tokens = []
            current_token_ids = []

            for token, offset, token_id in zip(tokens, offset_mapping.tolist(), input_ids.tolist()):
                start, end = offset

                # Skip special tokens ([CLS], [SEP], [PAD])
                if start == 0 and end == 0:
                    continue

                # Check for subwords (##) and group tokens into words
                if token.startswith("##"):
                    current_word += token[2:]
                    current_tokens.append(token)
                    current_token_ids.append(token_id)
                else:
                    # Save previous word
                    if current_word:
                        word_mapping.append((current_word, current_tokens, current_token_ids))
                    # Start a new word
                    current_word = token
                    current_tokens = [token]
                    current_token_ids = [token_id]

            # Save the last word
            if current_word:
                word_mapping.append((current_word, current_tokens, current_token_ids))

            word_level_timestamp_path = os.path.join(word_level_dir, row['uid'] + '.csv')

            # Read the word level timestamps
            df_word_level = pd.read_csv(word_level_timestamp_path)
            # Columns pandas_word_level = pd.DataFrame(columns=['word', 'start', 'end', 'probability'])
            words = []
            for index, data in df_word_level.iterrows():
                words.append((data['word'], data['start'], data['end']))

            idx_probs = 0
            act_word = ''

            idx_att = 0
            idx_start_att = 0

            idx_start_map = 0
            idx_map = 0

            n_audio_segments = 0

            # Print results
            for word, tokens, token_ids in word_mapping:
                # print(f"Word: {word}, Tokens: {tokens}, Token IDs: {token_ids}")
                cleaned_word = word.replace('Ġ', '')
                act_word += cleaned_word.replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower()

                print(f"Word: {word}, Tokens: {tokens}, Token IDs: {token_ids}")
                print(f"Act Word: {act_word}")
                if idx_probs < len(words):
                    # Check if words[idx_probs][0] is a string before printing
                    if isinstance(words[idx_probs][0], str):
                        print(f"Expected Word: {words[idx_probs][0].replace('Ġ', '').replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower()}")

                if word.strip() in ['.', ',', '?', '!', ';', 'Ġ','Ġ.', 'Ġ,', 'Ġ?', 'Ġ!', 'Ġ;', 'Ġ...', '...']:    # Ensure only real punctuation
                    if idx_probs > 0:  # Avoid index error
                        start = words[idx_probs-1][2]  # Get last word's end time
                    else:
                        start = 0  # Default to 0 if first word
                    end = words[idx_probs][1] if idx_probs < len(words) else None  # Safe check

                    start_segment = math.floor(start * segment_length)
                    end_segment = math.ceil(end * segment_length if end is not None else last_hidden_states_audio.shape[0])
                    print(f"FOUND PUNCTUATION: {word}, Start: {start}, End: {end}, Start Segment: {start_segment}, End Segment: {end_segment}")
                    print("Token IDs:")
                    for idx in range(idx_start_map, idx_map + 1):
                        print(f"{idx}: {word_mapping[idx]}")
                        print('------------------------------------------')

                    for idx in range(idx_start_att, idx_att + len(token_ids)):
                        n_audio_segments += 1

                        if end_segment - start_segment < 3:
                            start_segment = max(0, start_segment - 2)
                            end_segment = min(last_hidden_states_audio.shape[0], end_segment + 2)

                        audio_features_segment = last_hidden_states_audio[start_segment:end_segment]
                        processed_audio_tensor[idx + 1] = torch.clamp(audio_features_segment.mean(dim=0), min=-1e3, max=1e3)

                    idx_start_att = idx_att + len(token_ids)
                    idx_start_map = idx_map + 1



                if idx_probs < len(words) and isinstance(words[idx_probs][0], str) and act_word == words[idx_probs][0].replace('Ġ', '').replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower():

                        start = words[idx_probs][1]
                        end = words[idx_probs][2]

                        start_segment = math.floor(start * segment_length)
                        end_segment = math.ceil(end * segment_length if end is not None else last_hidden_states_audio.shape[0])

                        print(f"FOUND WORD: {act_word}, Start: {start}, End: {end}, Start Segment: {start_segment}, End Segment: {end_segment}")
                        print("Token IDs:")
                        for idx in range(idx_start_map, idx_map + 1):
                            print(f"{idx}: {word_mapping[idx]}")
                            print('------------------------------------------')

                        for idx in range(idx_start_att, idx_att + len(token_ids)):
                            n_audio_segments += 1

                            if end_segment - start_segment < 3:
                                start_segment = max(0, start_segment - 2)
                                end_segment = min(last_hidden_states_audio.shape[0], end_segment + 2)


                            audio_features_segment = last_hidden_states_audio[start_segment:end_segment]
                            processed_audio_tensor[idx + 1] = torch.clamp(audio_features_segment.mean(dim=0), min=-1e3, max=1e3)

                        idx_probs += 1
                        act_word = ''
                        idx_start_att = idx_att + len(token_ids)
                        idx_start_map = idx_map + 1

                idx_att += len(token_ids)
                idx_map += 1

            if idx_probs < len(words) and isinstance(words[idx_probs][0], str) and act_word in words[idx_probs][0].replace('Ġ', '').replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower():
                start = words[idx_probs][1]
                end = words[idx_probs][2]

                start_segment = math.floor(start * segment_length)
                end_segment = math.ceil(end * segment_length if end is not None else last_hidden_states_audio.shape[0])

                print(f"FOUND WORD: {act_word}, Start: {start}, End: {end}, Start Segment: {start_segment}, End Segment: {end_segment}")
                print("Token IDs:")
                for idx in range(idx_start_map, idx_map):
                    print(f"{idx}: {word_mapping[idx]}")
                    print('------------------------------------------')

                for idx in range(idx_start_att, idx_att):
                    n_audio_segments += 1

                    if end_segment - start_segment < 3:
                        start_segment = max(0, start_segment - 2)
                        end_segment = min(last_hidden_states_audio.shape[0], end_segment + 2)


                    audio_features_segment = last_hidden_states_audio[start_segment:end_segment]
                    processed_audio_tensor[idx + 1] = torch.clamp(audio_features_segment.mean(dim=0), min=-1e3, max=1e3)

                idx_probs += 1
                act_word = ''
                idx_start_att = idx_att + len(token_ids)
                idx_start_map = idx_map + 1


            print(f"Number of audio segments: {n_audio_segments}")
            # See inputs numbers and compare with the number of audio segments, separate with PAD tokens, eclusding them
            #total_tokens = torch.sum(inputs_text['input_ids'][0] != 0).item()
            total_tokens = torch.sum(inputs_text['attention_mask'][0]).item()
            print(f"Total tokens: {total_tokens}")
            if n_audio_segments + 2 != total_tokens:
                print(f"ERROR in {row['uid']}: Number of audio segments ({n_audio_segments}) does not match the number of tokens ({total_tokens})")
                print(f"Completed audios: {completed_audios}")
                return -1

            if torch.isnan(processed_audio_tensor).any():
                print(f"ERROR in {row['uid']}: NaN values in processed_audio_tensor")
                print(f"Completed audios: {completed_audios}")
                return -1

            torch.save(processed_audio_tensor, os.path.join(embeddings_dir, row['uid'] + textual_model_data + pauses_data + audio_model_data + '.pt'))


            completed_audios += 1

        print(f"------------------------------------------")
        print(f"CORRECTLY PROCESSED ALL AUDIOS")
        print(f"Completed audios: {completed_audios}")

preprocess_text()
