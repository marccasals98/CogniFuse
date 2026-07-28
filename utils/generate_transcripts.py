# Imports
# ---------------------------------------------------------------------
import argparse
import logging
import torch
import datetime
import os
import pandas as pd

import whisper
# ---------------------------------------------------------------------

#region Logging
# ---------------------------------------------------------------------
# Set logging config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter(
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = '%y-%m-%d %H:%M:%S',
    )

# Set a logging stream handler
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setLevel(logging.INFO)
logger_stream_handler.setFormatter(logger_formatter)

# Add handlers
logger.addHandler(logger_stream_handler)
#endregion
# ---------------------------------------------------------------------

# Classes
# ---------------------------------------------------------------------
class TranscriptionsGenerator:

    def __init__(self, params):
        
        self.params = params
        self.generate_dump_files_folder()
        self.set_log_file_handler()
        self.set_device()
        
        # in this version we are going to use the official transcripts and just fill the null transcripts with "..."
        self.transcripts_folder = "/home/usuaris/veussd/federico.costa/datasets/msp_podcast_2025/Transcripts"

    
    def generate_dump_files_folder(self):

        self.start_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y_%m_%d_%H_%M_%S_%f')
        self.dump_files_folder = os.path.join(self.params.dump_files_folder, self.start_datetime)
        if not os.path.exists(self.dump_files_folder): os.makedirs(self.dump_files_folder)

    
    def set_log_file_handler(self, log_file_name = "log.txt"):

        '''Set a logging file handler.'''
        
        logger_file_name = log_file_name
        logger_file_path = os.path.join(self.dump_files_folder, logger_file_name)
        logger_file_handler = logging.FileHandler(logger_file_path, mode = 'w')
        logger_file_handler.setLevel(logging.INFO) # TODO set the file handler level as a input param
        logger_file_handler.setFormatter(logger_formatter)

        logger.addHandler(logger_file_handler)

    
    def set_device(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        if self.device == "cuda":
            logger.info(f"{torch.cuda.device_count()} GPUs available.")

    
    def get_audio_filenames_from_labels(self, labels_tsv_path):

        df_labels = pd.read_csv(labels_tsv_path, sep = "\t")
        filenames = list(df_labels["filename"])

        return filenames
    
    
    def get_audio_filenames(self):

        train_audio_filenames = self.get_audio_filenames_from_labels(
            labels_tsv_path = self.params.train_labels_path,
        )

        dev_audio_filenames = self.get_audio_filenames_from_labels(
            labels_tsv_path = self.params.validation_labels_path,
        )

        test_audio_filenames = self.get_audio_filenames_from_labels(
            labels_tsv_path = self.params.test_labels_path,
        )

        self.audio_filenames = train_audio_filenames + dev_audio_filenames + test_audio_filenames
        self.files_partition = ["train"] * len(train_audio_filenames) + ["validation"] * len(dev_audio_filenames) + ["test"] * len(test_audio_filenames)
    
    
    def get_audio_paths(self):
        
        self.audio_paths = [
            os.path.join(self.params.audio_files_folder, audio_filename) for audio_filename in self.audio_filenames
        ]

    
    def get_transcript_filenames(self):

        self.transcript_filenames = [
            audio_filename.replace(".wav", ".txt") for audio_filename in self.audio_filenames
        ]

    
    def get_transcript_paths(self):

        self.transcript_paths = [
            os.path.join(self.transcripts_folder, audio_filename.replace(".wav", ".txt")) for audio_filename in self.audio_filenames
        ]

    
    def generate_transcripts(self):

        logger.info("Generating transcripts...")

        self.generated_transcripts = []

        for transcript_path, file_partition in zip(self.transcript_paths, self.files_partition):
            
            if file_partition == "test":
                # HACK test audios don't have transcripts.
                # We will fill these transcripts with some method ("..." in this version)
                transcript = []
            else:
                with open(transcript_path, 'r') as transcript_file:
                    transcript = transcript_file.readlines()

            if len(transcript) == 0:
                # in this case the transcript is empty
                transcript = "..."    
            elif len(transcript) == 1: 
                transcript = transcript[0]
            else: 
                raise Exception(f"Problems with the following transcription: {transcript_path}")
            
            self.generated_transcripts.append(transcript)
        
        logger.info("Done!")
    

    def generate_info(self):

        logger.info(f"Computing transcripts information...")

        df_info = pd.DataFrame(
            {
                "transcript_filename": self.transcript_filenames,
                "partition": self.files_partition,
            }
        )

        df_info["null_transcript"] = [True if transcript == "..." else False for transcript in self.generated_transcripts]

        logger.info(f"Audios distribution:")
        logger.info(f"{pd.Series(self.files_partition).value_counts(dropna = False)}")

        logger.info(f"Null transcripts filled with '...':")
        logger.info(f"{df_info.groupby('partition')['null_transcript'].value_counts(normalize=True)*100}")
        
        logger.info(f"Done!")

    
    def save_generated_transcripts(self):

        logger.info(f"Saving {len(self.transcript_filenames)} transcripts into {self.dump_files_folder}...")

        dump_transcripts_folder = os.path.join(self.dump_files_folder, "transcripts")
        if not os.path.exists(dump_transcripts_folder): os.makedirs(dump_transcripts_folder)
        
        for transcript_filename, generated_transcript in zip(self.transcript_filenames, self.generated_transcripts):
            try:
                with open(os.path.join(dump_transcripts_folder, transcript_filename), 'w') as f:
                    f.write(generated_transcript)
                    f.close()
            except Exception as e:
                logger.info(f"Failed to save file {transcript_filename} with transcription {generated_transcript}")
                logger.info(f"Error: {e}")
        
        logger.info("Done!")
    
    
    def main(self):
        
        self.get_audio_filenames()
        self.get_audio_paths()
        self.get_transcript_filenames()
        self.get_transcript_paths()
        self.generate_transcripts()
        self.generate_info()
        self.save_generated_transcripts()


class ArgsParser:

    def __init__(self):
        
        self.initialize_parser()

    
    def initialize_parser(self):

        self.parser = argparse.ArgumentParser(
            description = 'Takes the MSP-Podcast training, validation and test audios and generates the corresponding transcripts. \
                This version uses the official transcripts and fills with ... the null transcripts.', 
            )


    def add_parser_args(self):
        
        self.parser.add_argument(
            'audio_files_folder', 
            type = str, 
            help = 'Folder containing the audios to transcript.',
            )

        self.parser.add_argument(
            'train_labels_path', 
            type = str, 
            help = 'Train labels used for searching the corresponding audios and generate the transcripts.',
            )

        self.parser.add_argument(
            'validation_labels_path', 
            type = str, 
            help = 'Validation labels used for searching the corresponding audios and generate the transcripts.',
            )

        self.parser.add_argument(
            'test_labels_path', 
            type = str, 
            help = 'Test labels used for searching the corresponding audios and generate the transcripts.',
            )

        self.parser.add_argument(
            '--dump_files_folder', 
            type = str, 
            default = './files/', 
            help = 'Folder where we want to dump the file with the transcriptions and logs.',
            )
        

    def main(self):

        self.add_parser_args()
        self.arguments = self.parser.parse_args()

# ---------------------------------------------------------------------

if __name__=="__main__":

    args_parser = ArgsParser()
    args_parser.main()
    input_parameters = args_parser.arguments

    transcriptor = TranscriptionsGenerator(input_parameters)
    transcriptor.main()