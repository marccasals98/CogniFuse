# Imports
# ---------------------------------------------------------------------
import argparse
import os
import datetime
import pandas as pd
import logging
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
class TrainingLabelsGenerator:

    def __init__(self, params):
        
        self.params = params
        self.start_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y_%m_%d_%H_%M_%S_%f')
        self.set_log_file_handler()
        
        # We are only considering these classes for the final labels
        self.considered_classes = ["n", "h", "a", "s", "c", "u", "d", "f"]


    def set_log_file_handler(self, log_file_name = "log.txt"):

        '''Set a logging file handler.'''

        self.params.log_file_folder = os.path.join(self.params.dump_files_folder, self.start_datetime)
        
        if not os.path.exists(self.params.log_file_folder):
            os.makedirs(self.params.log_file_folder)
        
        logger_file_name = log_file_name
        logger_file_path = os.path.join(self.params.log_file_folder, logger_file_name)
        logger_file_handler = logging.FileHandler(logger_file_path, mode = 'w')
        logger_file_handler.setLevel(logging.INFO) # TODO set the file handler level as a input param
        logger_file_handler.setFormatter(logger_formatter)

        logger.addHandler(logger_file_handler) 


    def load_labels(self):

        '''Load files and labels information.'''

        logger.info("Loading info from labels file...")

        # Load de data
        self.labels_df = pd.read_csv(self.params.labels_file_path, sep = ",")

        # Lowercase the columns
        self.labels_df.columns = [col.lower() for col in self.labels_df]

        # Use only mandatory columns
        cols_to_keep = ["filename", "emoclass", "split_set"]
        self.labels_df = self.labels_df[cols_to_keep]
        self.labels_df.rename(columns = {"split_set": "partition"}, inplace=True)
            
        # Null processing
        self.labels_df.fillna({"filename": "null_value"}, inplace=True)
        self.labels_df.fillna({"emoclass": "null_value"}, inplace=True)
        self.labels_df.fillna({"partition": "null_value"}, inplace=True)
        
        filename_nulls = (self.labels_df["filename"] == "null_value").sum()
        emoclass_nulls = (self.labels_df["emoclass"] == "null_value").sum()
        partition_nulls = (self.labels_df["partition"] == "null_value").sum()
        logger.info(f"{filename_nulls} filename nulls")
        logger.info(f"{emoclass_nulls} emoclass nulls")
        logger.info(f"{partition_nulls} partition nulls")

        # Values cleaning
        self.labels_df["filename"] = self.labels_df["filename"].str.strip()
        self.labels_df["emoclass"] = self.labels_df["emoclass"].str.strip()
        self.labels_df["partition"] = self.labels_df["partition"].str.strip()
        self.labels_df["emoclass"] = self.labels_df["emoclass"].str.lower()
        self.labels_df["partition"] = self.labels_df["partition"].str.lower()

        # Classes filtering
        logger.info("Classes distribution before filtering:")
        logger.info(self.labels_df['emoclass'].value_counts(dropna=False))

        classes_filter = self.labels_df["emoclass"].isin(self.considered_classes)
        self.labels_df = self.labels_df[classes_filter]
        
        logger.info("Classes distribution after filtering:")
        logger.info(self.labels_df['emoclass'].value_counts(dropna=False))

        logger.info("Partition distribution:")
        logger.info(self.labels_df['partition'].value_counts(dropna=False))
        
        logger.info(f"Labels has {len(self.labels_df)} files.")

        logger.info("Labels loaded.")

    
    def load_partitions_info(self):

        '''Load each audio partition information (train, dev or test).'''

        logger.info("Loading info from partitions file...")

        # Load de data
        self.partitions_df = pd.read_csv(self.params.partitions_file_path, sep = ";", header = None)

        self.partitions_df.columns = ["partition", "filename"]

        # Null processing
        self.partitions_df.fillna({"partition": "null_value"}, inplace=True)
        self.partitions_df.fillna({"filename": "null_value"}, inplace=True)

        partition_nulls = (self.partitions_df["partition"] == "null_value").sum()
        filename_nulls = (self.partitions_df["filename"] == "null_value").sum()

        logger.info(f"{partition_nulls} partition nulls")
        logger.info(f"{filename_nulls} filename nulls")

        # Values cleaning
        self.partitions_df["partition"] = self.partitions_df["partition"].str.strip()
        self.partitions_df["filename"] = self.partitions_df["filename"].str.strip()
        self.partitions_df["partition"] = self.partitions_df["partition"].str.lower()

        logger.info(f"Partitions has {len(self.partitions_df)} lines.")
        logger.info("Partitions info loaded.")


    def merge_files(self):
        """
        Merge partitions and labels files information.
        Partitions has train, dev and test information, while labels only has train and dev.
        """

        logger.info(f"Merging partitions and labels info...")

        self.merged_df = pd.merge(
            self.partitions_df,
            self.labels_df, 
            how = "left", 
            on = "filename",
            suffixes = ["_partitions_df", "_labels_df"],
            indicator = True,
        )

        logger.info(f"{self.merged_df.iloc[0]}")
 
        # We are only keeping those files with a label class (train or dev), or those that are for test
        filter_1 = self.merged_df["emoclass"].isin(self.considered_classes)
        filter_2 = self.merged_df["partition_partitions_df"].isin(["test3"])
        self.merged_df = self.merged_df[filter_1 | filter_2]

        logger.info("Merge checks (left is partitions file and right is labels file):")
        logger.info(self.merged_df["_merge"].value_counts())

        logger.info(f"Files merged.")


    def generate_partition_labels(self, partition, final_cols, dump_files_folder, dump_file_name):

        logger.info(f"Generating {partition} labels...")

        filter = self.merged_df["partition_partitions_df"] == partition
        self.filtered_labels = self.merged_df[filter]
        self.filtered_labels = self.filtered_labels[final_cols]

        lines_to_write = list(zip(*[self.filtered_labels[col].values for col in self.filtered_labels.columns]))
        lines_to_write = ["\t".join(line_elements) for line_elements in lines_to_write]

        if not os.path.exists(dump_files_folder):
            os.makedirs(dump_files_folder)
        dump_path = os.path.join(dump_files_folder, dump_file_name)
        
        with open(dump_path, 'w') as f:
            # Write header
            f.write("\t".join(final_cols))
            f.write('\n')
            # Write rest of lines
            for line_to_write in lines_to_write:
                f.write(line_to_write)
                f.write('\n')
            f.close()

        logger.info(f"Saving {partition} labels into: {dump_path}...")
        logger.info(f"{partition} labels generated: {len(lines_to_write)} total lines.")
  

    def generate_labels(self):

        labels_folder = os.path.join(self.params.dump_files_folder, self.start_datetime)

        self.generate_partition_labels(
            partition = "train", 
            final_cols = ["filename", "emoclass"], 
            dump_files_folder = labels_folder, 
            dump_file_name = "training_labels.tsv",
        )

        self.generate_partition_labels(
            partition = "development", 
            final_cols = ["filename", "emoclass"], 
            dump_files_folder = labels_folder, 
            dump_file_name = "development_labels.tsv",
        )

        self.generate_partition_labels(
            partition = "test3", 
            final_cols = ["filename"], 
            dump_files_folder = labels_folder, 
            dump_file_name = "test_labels.tsv",
        )


    def main(self):

        # Load MSP-Podcast dataset labels
        self.load_labels()

        # Load MSP-Podcast partitions info
        self.load_partitions_info()

        # Merge both files
        self.merge_files()

        # Generate train, dev and test labels files
        self.generate_labels()


class ArgsParser:

    def __init__(self):
        
        self.initialize_parser()

    
    def initialize_parser(self):

        self.parser = argparse.ArgumentParser(
            description = 'Takes the MSP-Podcast labels and generate labels for training a SER classifier.', 
            )


    def add_parser_args(self):

        self.parser.add_argument(
            'labels_file_path',
            type = str, 
            help = 'Path of the csv file containing the labels information (labels_consensus.csv file).',
            )

        self.parser.add_argument(
            'partitions_file_path',
            type = str, 
            help = 'Path of the txt file containing the partitions information (Partitions.txt file).',
            )
        
        self.parser.add_argument(
            '--dump_files_folder', 
            type = str, 
            default = './labels_generator/files/', 
            help = 'Folder where we want to dump the file with the labels and logs.',
            )
        

    def main(self):

        self.add_parser_args()
        self.arguments = self.parser.parse_args()

# ---------------------------------------------------------------------

if __name__=="__main__":

    args_parser = ArgsParser()
    args_parser.main()
    input_parameters = args_parser.arguments

    labels_generator = TrainingLabelsGenerator(input_parameters)
    labels_generator.main()
    














