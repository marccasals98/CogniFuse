import whisper
import os
import pandas as pd
import torch
import random
import re

def remove_non_english(text):
    return re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)

model = whisper.load_model("turbo")


root_path = '/dataset/diagnosis/train/audio/'

diagnosis = ['ad', 'cn']
textual_data = '/dataset/diagnosis/train/text_transcriptions.csv'


def preprocess_whisper():

    df = pd.DataFrame(columns=['uid', 'diagno', 'transcription', 'transcription_pause', 'probablities'])

    for diagno in diagnosis:

        diagno_path = os.path.join(root_path, diagno)

        for file in os.listdir(diagno_path):

            if file.endswith(".wav"):
                print('Processing:', file)

                audio_path = os.path.join(diagno_path, file)

                word_level_path = audio_path.replace('.wav', '.csv').replace('audio', 'text')
                segmentation_path = audio_path.replace('.wav', '.csv').replace('audio', 'segmentation')

                excluding_times = []

                if os.path.exists(segmentation_path):
                    df_segmentation = pd.read_csv(segmentation_path)
                    df_segmentation = df_segmentation[df_segmentation['speaker'] == 'INV']
                    for segment in df_segmentation.iterrows():
                        excluding_times += [(segment[1]['begin']/1000, segment[1]['end']/1000)]

                idx_exclude = 0
                result = model.transcribe(audio_path, word_timestamps=True)

                probs = []
                print('Excluding times:', excluding_times)

                transcription = ''
                transcription_pauses = ''
                prev_start = 0.0

                pandas_word_level = pd.DataFrame(columns=['word', 'start', 'end', 'probability'])

                for segment in result['segments']:
                    # Print words in segment
                    for word in segment['words']:

                        if idx_exclude < len(excluding_times) and word['start'] >= excluding_times[idx_exclude][1]:
                            idx_exclude += 1

                        if idx_exclude >= len(excluding_times) or word['end'] < excluding_times[idx_exclude][0]:
                            transcription_pauses += word['word']
                            transcription += word['word']
                            clean_word = remove_non_english(word['word'].replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower())
                            if clean_word != '':
                                pandas_word_level = pandas_word_level._append({'word': clean_word, 'start': word['start'], 'end': word['end'], 'probability': word['probability']}, ignore_index=True)
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

                pandas_word_level.to_csv(word_level_path, index=False)

                df = df._append({'uid': file.replace('.wav', ''), 'diagno': diagno, 'transcription': remove_non_english(transcription), 'transcription_pause': remove_non_english(transcription_pauses), 'probablities': probs}, ignore_index=True)

    df.to_csv(textual_data, index=False)

preprocess_whisper()