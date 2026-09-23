import whisper
import os
import pandas as pd
import torch
import random
import re

import unicodedata

def clean_text(text):
    """
    Clean the text by removing unwanted characters and normalizing it.
    Now it is adapted to Spanish and Catalan.
    We need to preserve characters such as accents, ñ, ü...
    """
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"[^\w\s.,!?'\"¿¡·’-]", "", text)

model = whisper.load_model("turbo")


dataset_path = '/home/usuaris/veussd/marc.casals/datasets/WAB_samples'
root_path = os.path.join(dataset_path, 'WAB_samples')
output_path = os.path.join(dataset_path, 'preprocessing')
word_level_dir = os.path.join(output_path, 'words')
segmentation_dir = os.path.join(dataset_path, 'segmentation')
textual_data = os.path.join(output_path, 'transcriptions.csv')


def preprocess_whisper():

    os.makedirs(word_level_dir, exist_ok=True)

    recordings = []

    for file in sorted(os.listdir(root_path)):

        # Read the .wav or .mp3 file and transcribe it using the Whisper model
        if file.lower().endswith((".wav", ".mp3", ".alac")) and os.path.isfile(os.path.join(root_path, file)):
            print('Processing:', file)

            audio_path = os.path.join(root_path, file)
            uid = os.path.splitext(file)[0]

            word_level_path = os.path.join(word_level_dir, uid + '.csv')
            segmentation_path = os.path.join(segmentation_dir, uid + '.csv')

            excluding_times = []

            if os.path.exists(segmentation_path):
                df_segmentation = pd.read_csv(segmentation_path)
                df_segmentation = df_segmentation[df_segmentation['speaker'] == 'INV']
                for segment in df_segmentation.iterrows():
                    excluding_times += [(segment[1]['begin']/1000, segment[1]['end']/1000)]

            idx_exclude = 0
            result = model.transcribe(audio_path,

                                    task="transcribe",
                                    word_timestamps=True)

            probs = []
            print('Excluding times:', excluding_times)

            transcription = ''
            transcription_pauses = ''
            prev_start = 0.0

            word_rows = []

            for segment in result['segments']:
                # Print words in segment
                for word in segment['words']:

                    if idx_exclude < len(excluding_times) and word['start'] >= excluding_times[idx_exclude][1]:
                        idx_exclude += 1

                    if idx_exclude >= len(excluding_times) or word['end'] < excluding_times[idx_exclude][0]:
                        transcription_pauses += word['word']
                        transcription += word['word']
                        clean_word = clean_text(word['word']).strip()
                        if clean_word != '':
                            word_rows.append({'word': clean_word, 'start': word['start'], 'end': word['end'], 'probability': word['probability']})
                        probs += [(clean_word, word['probability'])]


                        if prev_start > 0.0:
                            pause = word['start'] - prev_start

                            if pause > 2:
                                transcription_pauses += ' ...'
                            elif pause > 1:
                                transcription_pauses += ' .'
                            elif pause > 0.5:
                                transcription_pauses += ' ,'

                        prev_start = word['end']
                    else:
                        print('Excluding word:', word)

                    if idx_exclude < len(excluding_times) and word['end'] >= excluding_times[idx_exclude][1]:
                        idx_exclude += 1

            print('Result:', result['text'])
            print('Transcription:', transcription)
            print('Transcription pauses:', transcription_pauses)
            print('Probs:', probs)

            pandas_word_level = pd.DataFrame(word_rows, columns=['word', 'start', 'end', 'probability'])
            pandas_word_level.to_csv(word_level_path, index=False)

            recordings.append({'uid': uid, 'transcription': clean_text(transcription), 'transcription_pause': clean_text(transcription_pauses), 'probablities': probs})

    df = pd.DataFrame(recordings, columns=['uid', 'transcription', 'transcription_pause', 'probablities'])
    df.to_csv(textual_data, index=False)

preprocess_whisper()
