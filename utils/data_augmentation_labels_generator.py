# Imports
# ---------------------------------------------------------------------
import argparse
import os
import pandas as pd
import logging
import datetime
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
class AugmentationLabelsGenerator:

    def __init__(self, params):
        
        self.params = params
        self.start_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y_%m_%d_%H_%M_%S_%f')
        self.set_log_file_handler()


    def set_log_file_handler(self):

        '''Set a logging file handler.'''

        self.params.log_file_folder = os.path.join(self.params.dump_files_folder, self.start_datetime)

        if not os.path.exists(self.params.log_file_folder):
            os.makedirs(self.params.log_file_folder)
        
        logger_file_name = self.params.log_file_name
        logger_file_path = os.path.join(self.params.log_file_folder, logger_file_name)
        logger_file_handler = logging.FileHandler(logger_file_path, mode = 'w')
        logger_file_handler.setLevel(logging.INFO) # TODO set the file handler level as a input param
        logger_file_handler.setFormatter(logger_formatter)

        logger.addHandler(logger_file_handler) 

    
    def search_files(self, folder_path):

        lines_to_write = []

        logger.debug(f"Searching {self.params.valid_audio_formats} files in {folder_path}")
        
        finish_search = False
        for (dir_path, dir_names, file_names) in os.walk(folder_path):

                if self.params.verbose: logger.debug(f"Searching in {dir_path}")

                for file_name in file_names:
        
                    if file_name.split(".")[-1] in self.params.valid_audio_formats:
                        
                        path_to_write = os.path.join(dir_path, file_name)
                        lines_to_write.append(path_to_write)

                        if self.params.dump_max_lines != -1 and len(lines_to_write) >= self.params.dump_max_lines:
                            finish_search = True

                        if finish_search: break

                    if finish_search: break
                    
                if finish_search: break

        logger.debug(f"{len(lines_to_write)} files founded in {folder_path}")

        return lines_to_write
    
    
    def dump_paths(self, paths, dump_file_path):

        logger.info(f"Dumping {len(paths)} files paths into {dump_file_path}")
        
        with open(dump_file_path, 'w') as file:

            for line_to_write in paths:
                file.write(line_to_write)
                file.write('\n')

        logger.info(f"{len(paths)} files paths dumped in {dump_file_path}")
    
    
    def generate_rirs_labels(self):

        logger.info(f"Generating RIRs labels...")

        self.params.rirs_dump_file_folder = os.path.join(self.params.rirs_dump_file_folder, self.start_datetime)

        rirs_lines_to_write = []
        # Add header
        #rirs_lines_to_write.append("file_path")
        for dir in self.params.rirs_data_folders:
            new_lines = self.search_files(dir)
            rirs_lines_to_write = rirs_lines_to_write + new_lines
        
        dump_path = os.path.join(self.params.rirs_dump_file_folder, self.params.rirs_dump_file_name)
        if not os.path.exists(self.params.rirs_dump_file_folder):
            os.makedirs(self.params.rirs_dump_file_folder)

        self.dump_paths(rirs_lines_to_write, dump_path)

        logger.info(f"RIRs labels generated.")

    
    def generate_background_noises_labels(self):

        logger.info(f"Generating background noises labels...")

        self.params.noises_dump_file_folder = os.path.join(self.params.noises_dump_file_folder, self.start_datetime)

        noises_lines_to_write = []

        # Add header
        #noises_lines_to_write.append("file_path\tlabel")

        for dir in self.params.noises_data_folders:
            new_lines = self.search_files(dir)
            noises_lines_to_write = noises_lines_to_write + new_lines
        
        # HACK specific to my folders
        noises_lines_to_write_with_labels = []
        for line_num, line_to_write in enumerate(noises_lines_to_write):
            if "musan/noise/" in line_to_write or "RIRS_NOISES/pointsource_noises/" in line_to_write:
                noises_lines_to_write_with_labels.append(f"{line_to_write}\tnoise")
            elif "musan/music/" in line_to_write:
                noises_lines_to_write_with_labels.append(f"{line_to_write}\tmusic")
            elif "musan/speech/" in line_to_write:
                noises_lines_to_write_with_labels.append(f"{line_to_write}\tspeech")
            else:
                raise Exception(f'Case not considered when labelling background noise {line_to_write}')
        
        dump_path = os.path.join(self.params.noises_dump_file_folder, self.params.noises_dump_file_name)
        if not os.path.exists(self.params.noises_dump_file_folder):
            os.makedirs(self.params.noises_dump_file_folder)

        self.dump_paths(noises_lines_to_write_with_labels, dump_path)

        logger.info(f"Background noises labels generated.")

    
    def main(self):

        self.generate_rirs_labels()
        self.generate_background_noises_labels()


class ArgsParser:

    def __init__(self):
        
        self.initialize_parser()

    
    def initialize_parser(self):

        self.parser = argparse.ArgumentParser(
            description = 'Generates labels neccesary for speech augmentation.',
            )

    def add_parser_args(self):

        self.parser.add_argument(
            '--rirs_data_folders',
            nargs = '+',
            type = str, 
            help = 'Folder(s) containing the RIRs audio files we want to extract paths from.',
            )

        self.parser.add_argument(
            '--noises_data_folders',
            nargs = '+',
            type = str, 
            help = 'Folder(s) containing the background noises audio files we want to extract paths from.',
            )

        self.parser.add_argument(
            '--rirs_dump_file_folder', 
            type = str, 
            default = './data_augmentation_labels_generator/files/',
            help = 'Data folder where we want to dump the rirs paths file.',
            )
        
        self.parser.add_argument(
            '--rirs_dump_file_name', 
            type = str, 
            default = 'data_augmentation_rirs_labels.tsv', 
            help = 'Name of the file we want to dump rirs paths into.',
            )

        self.parser.add_argument(
            '--noises_dump_file_folder', 
            type = str, 
            default = './data_augmentation_labels_generator/files/',
            help = 'Data folder where we want to dump the background noises paths file.',
            )
        
        self.parser.add_argument(
            '--noises_dump_file_name', 
            type = str, 
            default = 'data_augmentation_noises_labels.tsv', 
            help = 'Name of the file we want to dump noises paths into.',
            )

        self.parser.add_argument(
            '--dump_files_folder', 
            type = str, 
            default = './labels_generator/files/', 
            help = 'Folder where we want to dump the file with the labels and logs.',
            )

        self.parser.add_argument(
            '--valid_audio_formats', 
            action = 'append',
            default = ['wav'],
            help = 'Audio files extension to search for.',
            )

        self.parser.add_argument(
            '--dump_max_lines', 
            type = int, 
            default = -1, 
            help = 'Max lines to dump from every folder search. Set to -1 if no limitation is wanted.',
            )

        self.parser.add_argument(
            "--verbose", 
            action = argparse.BooleanOptionalAction,
            default = False,
            help = "Increase output verbosity.",
            )
        

    def main(self):

        self.add_parser_args()
        self.arguments = self.parser.parse_args()

# ---------------------------------------------------------------------

if __name__=="__main__":

    args_parser = ArgsParser()
    args_parser.main()
    input_parameters = args_parser.arguments

    labels_generator = AugmentationLabelsGenerator(input_parameters)
    labels_generator.main()
    














