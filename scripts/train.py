# ---------------------------------------------------------------------
#region Imports

import argparse
import datetime
import logging
import numpy as np
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import f1_score
#from torchsummary import summary
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data import ADDataset
from model import Classifier
from loss import FocalLossCriterion
from utils import format_training_labels, generate_model_name, get_memory_info, pad_collate, get_waveforms_stats
from settings import TRAIN_DEFAULT_SETTINGS, LABELS_TO_IDS
#endregion

# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
#region Logging

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


# ---------------------------------------------------------------------
# Classes

class Trainer:

    def __init__(self, input_params):

        self.start_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H:%M:%S')
        if input_params.use_weights_and_biases: self.init_wandb(input_params)
        if input_params.number_classes == 6:
            logger.info("Assuming you are using Spanish MEAcorpus, EmoSPeech dataset.")
            logger.info("Using LABELS_TO_IDS_EMOTSPEECH dictionary for class mapping")
        self.set_device()
        self.set_random_seed(input_params)
        self.set_params(input_params)
        self.set_log_file_handler(logger_level = "info")
        self.load_data()
        self.load_network()
        self.load_loss_function()
        self.load_optimizer()
        self.initialize_training_variables()
        if self.params.use_weights_and_biases: self.config_wandb()


    def init_wandb(self, input_params):
        """
        Init a wandb project
        """

        # TODO fix this, it should be more general to other users
        self.wandb_run = wandb.init(
            project = "emotions_trains_2025",
            job_type = "training",
            entity = "upc-veu",
            dir = input_params.wandb_dir,
            resume = "allow",
            mode = "offline",
            )
        logger.info(f"wandb running online/offline: {self.wandb_run.settings.mode}")
        logger.info(f"dir for wandb init: {input_params.wandb_dir}")
        logger.info(f"Run id: {wandb.run.id}_{wandb.run.name}")

        # free memory
        #del wandb_run


    def set_device(self):
        """
        Set torch device.
        """

        logger.info('Setting device...')

        # Set device to GPU or CPU depending on what is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Running on {self.device} device.")

        if self.device == "cuda":
            self.gpus_count = torch.cuda.device_count()
            logger.info(f"{self.gpus_count} GPUs available.")
            # Batch size should be divisible by number of GPUs
        else:
            self.gpus_count = 0

        logger.info("Device setted.")



    def setup_distributed(self):
        """
        Setup distributed training environment.

        This method checks for distributed training environment variables and initializes
        PyTorch distributed process group if they are present.

        To run with DistributedDataParallel (DDP):

        Single-node multi-GPU (recommended):
            torchrun --nproc_per_node=NUM_GPUS train.py [args...]

        Example with 4 GPUs:
            torchrun --nproc_per_node=4 train.py --speech_feature_extractor WAV2VEC2_XLSR_300M ...

        Multi-node multi-GPU:
            # On each node:
            torchrun --nproc_per_node=NUM_GPUS --nnodes=NUM_NODES --node_rank=NODE_RANK \
                     --master_addr=MASTER_ADDR --master_port=MASTER_PORT train.py [args...]

        Without DDP (single GPU or CPU):
            python train.py [args...]

        Environment variables (set automatically by torchrun):
            - RANK: Global rank of the process
            - WORLD_SIZE: Total number of processes
            - LOCAL_RANK: Local rank on the current node
            - MASTER_ADDR: Address of rank 0 process
            - MASTER_PORT: Port of rank 0 process
        """

        # Check if distributed training is initialized
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))

            logger.info(f"Distributed training detected: rank={self.rank}, world_size={self.world_size}, local_rank={self.local_rank}")

            # Initialize the process group
            dist.init_process_group(backend='nccl')

            # Set device to the local rank
            torch.cuda.set_device(self.local_rank)
            self.device = f"cuda:{self.local_rank}"

            self.is_distributed = True
            self.is_main_process = (self.rank == 0)

            logger.info(f"Distributed training initialized on device {self.device}")
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.is_distributed = False
            self.is_main_process = True

            logger.info("No distributed training detected, running on single process")

    def set_random_seed(self, input_params):

        logger.info("Setting random seed...")

        random.seed(input_params.random_seed)
        np.random.seed(input_params.random_seed)

        torch.manual_seed(input_params.random_seed)
        torch.cuda.manual_seed(input_params.random_seed)
        torch.cuda.manual_seed_all(input_params.random_seed)  # if you are using multi-GPU.

        # sometimes using this in True yields worse results
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        logger.info(f"Random seed setted to {input_params.random_seed}.")


    def set_params(self, input_params):
        """
        Set Trainer class parameters.
        """

        logger.info("Setting params...")

        self.params = input_params

        self.params.model_architecture_name = f"{self.params.speech_feature_extractor}_{self.params.text_feature_extractor}_{self.params.speech_adapter}_{self.params.text_adapter}_{self.params.seq_to_seq_method}_{self.params.seq_to_one_method}"

        if self.params.use_weights_and_biases:
            self.params.model_name = generate_model_name(
                self.params,
                start_datetime = self.start_datetime,
                wandb_run_id = wandb.run.id,
                wandb_run_name = wandb.run.name
            )
        else:
            self.params.model_name = generate_model_name(
                self.params,
                start_datetime = self.start_datetime,
            )

        if self.params.load_checkpoint == True:
            self.load_checkpoint()
            self.load_checkpoint_params()
            # When we load checkpoint params, all input params are overwriten.
            # So we need to set load_checkpoint flag to True
            self.params.load_checkpoint = True
            # TODO here we could set a new max_epochs value

        logger.info(f"model_architecture_name: {self.params.model_architecture_name}")
        logger.info(f"model_name: {self.params.model_name}")
        logger.info("params setted.")


    def load_checkpoint(self):
        """
        Load trained model checkpoint to continue its training.
        """

        # Load checkpoint
        checkpoint_path = os.path.join(
            self.params.checkpoint_file_folder,
            self.params.checkpoint_file_name,
        )

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        self.checkpoint = torch.load(checkpoint_path, map_location = self.device)

        logger.info(f"Checkpoint loaded.")


    def load_checkpoint_params(self):
        """
        Load checkpoint original parameters.
        """

        logger.info(f"Loading checkpoint params...")

        self.params = self.checkpoint['settings']

        logger.info(f"Checkpoint params loaded.")


    def set_log_file_handler(self, logger_level = "info"):
        """
        Set a logging file handler.
        """

        if not os.path.exists(self.params.log_file_folder):
            os.makedirs(self.params.log_file_folder)

        if self.params.use_weights_and_biases:
            logger_file_name = f"{self.start_datetime}_{wandb.run.id}_{wandb.run.name}.log"
        else:
            logger_file_name = f"{self.start_datetime}.log"
        logger_file_name = logger_file_name.replace(':', '_').replace(' ', '_').replace('-', '_')

        logger_file_path = os.path.join(self.params.log_file_folder, logger_file_name)
        logger_file_handler = logging.FileHandler(logger_file_path, mode = 'w')

        # TODO set the file handler level as a input param
        if logger_level == "info":
            logger_file_handler.setLevel(logging.INFO)
        else:
            logger_file_handler.setLevel(logging.DEBUG)

        logger_file_handler.setFormatter(logger_formatter)

        logger.addHandler(logger_file_handler)


    def format_train_labels(self):

        return format_training_labels(
            labels_path = self.params.train_labels_path,
            labels_to_ids = LABELS_TO_IDS,
            prepend_directory = self.params.train_data_dir,
            header = True,
        )


    def format_validation_labels(self):

        return format_training_labels(
            labels_path = self.params.validation_labels_path,
            labels_to_ids = LABELS_TO_IDS,
            prepend_directory = self.params.validation_data_dir,
            header = True,
        )


    def format_labels(self):
        """
        Loads train and validation labels, formats them and returns (train_labels_lines, validation_labels_lines)
        """

        return self.format_train_labels(), self.format_validation_labels()


    def load_training_data(self):

        logger.info(f'Loading training data with labels from {self.params.train_labels_path}')

        # TODO delete this, it is used for normalizing, but each speech extractor should do the corresponding normalization
        #self.training_wav_mean, self.training_wav_std = get_waveforms_stats(train_labels_lines, self.params.sample_rate)

        # Instanciate a Dataset class
        training_dataset = ADDataset(
            input_parameters=self.params,
            audio_dir=self.params.train_data_dir,
            split="train",
            fold=1, # HACK: need to change it later
            target_classes=["lvPPA", "nfPPA", "svPPA"],
            ignore_labels=["exclude", "bvFTD"]
            )

        # To be used in the weighted loss
        if self.params.weighted_loss:
            self.training_dataset_classes_weights = training_dataset.get_classes_weights()
            self.training_dataset_classes_weights = torch.tensor(self.training_dataset_classes_weights).float().to(self.device)

        # Load DataLoader params
        if self.params.text_feature_extractor != 'NoneTextExtractor':
            data_loader_parameters = {
                'batch_size': self.params.training_batch_size,
                'shuffle': True,
                'num_workers': self.params.num_workers,
                'collate_fn': pad_collate,
                }
        else:
            data_loader_parameters = {
                'batch_size': self.params.training_batch_size,
                'shuffle': True,
                'num_workers': self.params.num_workers,
                }

        # TODO dont add to the class to get a lighter model?
        # Instanciate a DataLoader class
        self.training_generator = DataLoader(
            training_dataset,
            **data_loader_parameters,
            )

        del training_dataset

        logger.info("Data and labels loaded.")


    def set_evaluation_batch_size(self):
        # If evaluation is done using the full audio, batch size must be 1 because we will have different-size samples
        if self.params.evaluation_random_crop_secs == 0:
            self.params.evaluation_batch_size = 1


    def load_validation_data(self):

        logger.info(f'Loading data from {self.params.validation_labels_path}')

        # Instanciate a Dataset class
        validation_dataset = ADDataset(
            input_parameters=self.params,
            audio_dir=self.params.validation_data_dir,
            split="val",
            fold=1,
            target_classes=["lvPPA", "nfPPA", "svPPA"],
            ignore_labels=["exclude", "bvFTD"],
        )

        # If evaluation_type is total_length, batch size must be 1 because we will have different-size samples
        self.set_evaluation_batch_size()

        if self.params.text_feature_extractor != 'NoneTextExtractor':
            data_loader_parameters = {
                'batch_size': self.params.evaluation_batch_size,
                'shuffle': False,
                'num_workers': self.params.num_workers,
                'collate_fn': pad_collate,
                }
        else:
            data_loader_parameters = {
                'batch_size': self.params.evaluation_batch_size,
                'shuffle': False,
                'num_workers': self.params.num_workers,
                }

        # TODO dont add to the class to get a lighter model?
        # Instanciate a DataLoader class
        self.evaluating_generator = DataLoader(
            validation_dataset,
            **data_loader_parameters,
            )

        self.evaluation_total_batches = len(self.evaluating_generator)

        del validation_dataset

        logger.info("Data and labels loaded.")


    def load_data(self):

        self.load_training_data()
        self.load_validation_data()



    def load_checkpoint_network(self):

        logger.info(f"Loading checkpoint network...")

        # Try loading with module prefix (for DDP models), otherwise load directly
        try:
            self.net.load_state_dict(self.checkpoint['model'])
            logger.info(f"Checkpoint network loaded directly.")
        except RuntimeError:
            # If the model was saved with DDP, it has 'module.' prefix
            # Try loading into the module attribute
            try:
                self.net.module.load_state_dict(self.checkpoint['model'])
                logger.info(f"Checkpoint network loaded into module.")
            except AttributeError:
                # Model doesn't have module attribute, so remove 'module.' prefix from checkpoint
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in self.checkpoint['model'].items():
                    name = k.replace('module.', '')  # remove 'module.' prefix
                    new_state_dict[name] = v
                self.net.load_state_dict(new_state_dict)
                logger.info(f"Checkpoint network loaded after removing 'module.' prefix.")

        logger.info(f"Checkpoint network loaded.")


    def load_network(self):

        # Load the model (Neural Network)

        logger.info("Loading the network...")

        # Load model class
        self.net = Classifier(self.params, self.device)

        if self.params.load_checkpoint == True:
            self.load_checkpoint_network()

        # Assign model to device
        self.net.to(self.device)

        # Wrap with DistributedDataParallel if in distributed mode
        if self.is_distributed:
            logger.info("Wrapping model with DistributedDataParallel...")
            self.net = DDP(
                self.net,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False  # Set to True if you have unused parameters
            )
            logger.info("Model wrapped with DDP.")

        logger.info(self.net)

        # Use if parameter details are needed
        if False:
            self.total_trainable_params = 0
            parms_dict = {}
            logger.info(f"Detail of every trainable layer:")
            for name, parameter in self.net.named_parameters():

                layer_name = name.split(".")[1]
                if layer_name not in parms_dict.keys():
                    parms_dict[layer_name] = 0

                logger.debug(f"name: {name}, layer_name: {layer_name}")

                if not parameter.requires_grad:
                    continue
                trainable_params = parameter.numel()

                logger.debug(f"{name} is trainable with {parameter.numel()} parameters")

                parms_dict[layer_name] = parms_dict[layer_name] + trainable_params

                self.total_trainable_params += trainable_params

            logger.debug(f"Total trainable parameters per layer:")
            for layer_name in parms_dict.keys():
                logger.info(f"{layer_name}: {parms_dict[layer_name]}")

            #summary(self.net, (150, self.params.feature_extractor_output_vectors_dimension))

        self.total_params  = sum(p.numel() for p in self.net.parameters())
        self.total_trainable_params  = sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        logger.info(f"Network loaded, total_trainable_params: {self.total_trainable_params}, , total_params: {self.total_params}")


    def load_loss_function(self):

        logger.info("Loading the loss function...")

        if self.params.loss == "CrossEntropy":

            # The nn.CrossEntropyLoss() criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class

            if self.params.weighted_loss:
                logger.info("Using weighted loss function...")
                self.loss_function = nn.CrossEntropyLoss(
                    weight = self.training_dataset_classes_weights,
                )
            else:
                logger.info("Using unweighted loss function...")
                self.loss_function = nn.CrossEntropyLoss()

        elif self.params.loss == "FocalLoss":

            if self.params.weighted_loss:
                logger.info("Using weighted loss function...")
                self.loss_function = FocalLossCriterion(
                    gamma = 2,
                    weights = self.training_dataset_classes_weights,
                )
            else:
                logger.info("Using unweighted loss function...")
                self.loss_function = FocalLossCriterion(
                    gamma = 2,
                )

        else:
            raise Exception('No Loss choice found.')

        logger.info("Loss function loaded.")


    def load_checkpoint_optimizer(self):

        logger.info(f"Loading checkpoint optimizer...")

        self.optimizer.load_state_dict(self.checkpoint['optimizer'])

        logger.info(f"Checkpoint optimizer loaded.")


    def load_optimizer(self):

        logger.info("Loading the optimizer...")

        if self.params.optimizer == 'adam':
            self.optimizer = optim.Adam(
                #self.net.parameters(),
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.params.learning_rate,
                weight_decay=self.params.weight_decay,
                )
        if self.params.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                #self.net.parameters(),
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.params.learning_rate,
                weight_decay=self.params.weight_decay,
                )
        if self.params.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(
                #self.net.parameters(),
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.params.learning_rate,
                weight_decay=self.params.weight_decay,
                )
        if self.params.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                #self.net.parameters(),
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.params.learning_rate,
                weight_decay=self.params.weight_decay,
                )

        if self.params.load_checkpoint == True:
            self.load_checkpoint_optimizer()

        logger.info(f"Optimizer {self.params.optimizer} loaded.")


    def initialize_training_variables(self):

        logger.info("Initializing training variables...")

        if self.params.load_checkpoint == True:

            logger.info(f"Loading checkpoint training variables...")

            loaded_training_variables = self.checkpoint['training_variables']

            # HACK this can be refined, but we are going to continue training \
            # from the last epoch trained and from the first batch
            # (even if we may already trained with some batches in that epoch in the last training from the checkpoint).
            self.starting_epoch = loaded_training_variables['epoch']
            self.step = loaded_training_variables['step'] + 1
            self.validations_without_improvement = loaded_training_variables['validations_without_improvement']
            self.validations_without_improvement_or_opt_update = loaded_training_variables['validations_without_improvement_or_opt_update']
            self.early_stopping_flag = False
            self.train_loss = loaded_training_variables['train_loss']
            self.training_eval_metric = loaded_training_variables['training_eval_metric']
            self.validation_eval_metric = loaded_training_variables['validation_eval_metric']
            self.best_train_loss = loaded_training_variables['best_train_loss']
            self.best_model_train_loss = loaded_training_variables['best_model_train_loss']
            self.best_model_training_eval_metric = loaded_training_variables['best_model_training_eval_metric']
            self.best_model_validation_eval_metric = loaded_training_variables['best_model_validation_eval_metric']

            logger.info(f"Checkpoint training variables loaded.")
            logger.info(f"Training will start from:")
            logger.info(f"Epoch {self.starting_epoch}")
            logger.info(f"Step {self.step}")
            logger.info(f"validations_without_improvement {self.validations_without_improvement}")
            logger.info(f"validations_without_improvement_or_opt_update {self.validations_without_improvement_or_opt_update}")
            logger.info(f"Loss {self.train_loss:.3f}")
            logger.info(f"best_model_train_loss {self.best_model_train_loss:.3f}")
            logger.info(f"best_model_training_eval_metric {self.best_model_training_eval_metric:.3f}")
            logger.info(f"best_model_validation_eval_metric {self.best_model_validation_eval_metric:.3f}")

        else:
            self.starting_epoch = 0
            self.step = 0
            self.validations_without_improvement = 0
            self.validations_without_improvement_or_opt_update = 0
            self.early_stopping_flag = False
            self.train_loss = None
            self.training_eval_metric = 0.0
            self.validation_eval_metric = 0.0
            self.best_train_loss = np.inf
            self.best_model_train_loss = np.inf
            self.best_model_training_eval_metric = 0.0
            self.best_model_validation_eval_metric = 0.0

        self.total_batches = len(self.training_generator)

        logger.info("Training variables initialized.")


    def config_wandb(self):

        # 1 - Save the params
        self.wandb_config = vars(self.params)

        # 3 - Save additional params

        self.wandb_config["total_trainable_params"] = self.total_trainable_params
        self.wandb_config["gpus"] = self.gpus_count

        # 4 - Update the wandb config
        #wandb.config.update(self.wandb_config)
        self.wandb_run.config.update(self.wandb_config)


    def evaluate_training(self):

        logger.info(f"Evaluating training task...")

        with torch.no_grad():

            # Switch torch to evaluation mode
            self.net.eval()

            final_predictions, final_labels = torch.tensor([]).to("cpu"), torch.tensor([]).to("cpu")
            for batch_number, batch_data in enumerate(self.training_generator):

                if batch_number % 1000 == 0:
                    logger.info(f"Evaluating training task batch {batch_number} of {len(self.training_generator)}...")

                if self.params.text_feature_extractor != 'NoneTextExtractor':
                    input, label, transcription_tokens_padded, transcription_tokens_mask = batch_data
                else:
                    input, label = batch_data

                # Assign batch data to device
                if self.params.text_feature_extractor != 'NoneTextExtractor':
                    transcription_tokens_padded, transcription_tokens_mask = transcription_tokens_padded.long().to(self.device), transcription_tokens_mask.long().to(self.device)
                input, label = input.float().to(self.device), label.long().to(self.device)

                if batch_number == 0: logger.info(f"input.size(): {input.size()}")

                # Calculate prediction and loss
                if self.params.text_feature_extractor != 'NoneTextExtractor':
                    prediction  = self.net(
                        input_tensor = input,
                        transcription_tokens_padded = transcription_tokens_padded,
                        transcription_tokens_mask = transcription_tokens_mask,
                        )
                else:
                    prediction  = self.net(input_tensor = input)
                prediction = prediction.to("cpu")
                label = label.to("cpu")

                final_predictions = torch.cat(tensors = (final_predictions, prediction))
                final_labels = torch.cat(tensors = (final_labels, label))

            metric_score = f1_score(
                y_true = np.argmax(final_predictions, axis = 1),
                y_pred = final_labels,
                average='macro',
                )

            self.training_eval_metric = metric_score

            del final_predictions
            del final_labels

        # Return to torch training mode
        self.net.train()

        logger.info(f"Training task evaluated.")
        logger.info(f"F1-score (macro) on training set: {self.training_eval_metric:.3f}")


    def evaluate_validation(self):

        logger.info(f"Evaluating validation task...")

        with torch.no_grad():

            # Switch torch to evaluation mode
            self.net.eval()

            final_predictions, final_labels = torch.tensor([]).to("cpu"), torch.tensor([]).to("cpu")
            for batch_number, batch_data in enumerate(self.evaluating_generator):

                if batch_number % 1000 == 0:
                    logger.info(f"Evaluating validation task batch {batch_number} of {len(self.evaluating_generator)}...")

                if self.params.text_feature_extractor != 'NoneTextExtractor':
                    input, label, transcription_tokens_padded, transcription_tokens_mask = batch_data
                else:
                    input, label = batch_data

                # Assign batch data to device
                if self.params.text_feature_extractor != 'NoneTextExtractor':
                    transcription_tokens_padded, transcription_tokens_mask = transcription_tokens_padded.long().to("cpu"), transcription_tokens_mask.long().to("cpu")
                input, label = input.float().to("cpu"), label.long().to("cpu")

                if batch_number == 0: logger.info(f"input.size(): {input.size()}")

                # Calculate prediction and loss
                if self.params.text_feature_extractor != 'NoneTextExtractor':
                    prediction  = self.net(
                        input_tensor = input,
                        transcription_tokens_padded = transcription_tokens_padded,
                        transcription_tokens_mask = transcription_tokens_mask,
                        )
                else:
                    prediction  = self.net(input_tensor = input)
                prediction = prediction.to("cpu")
                label = label.to("cpu")

                final_predictions = torch.cat(tensors = (final_predictions, prediction))
                final_labels = torch.cat(tensors = (final_labels, label))

            metric_score = f1_score(
                y_true = np.argmax(final_predictions, axis = 1),
                y_pred = final_labels,
                average='macro',
                )

            self.validation_eval_metric = metric_score

            del final_predictions
            del final_labels

        # Return to training mode
        self.net.train()

        logger.info(f"Validation task evaluated.")
        logger.info(f"F1-score (macro) on validation set: {self.validation_eval_metric:.3f}")


    def evaluate(self):

        self.evaluate_training()
        self.evaluate_validation()


    def save_model(self):

        '''Function to save the model info and optimizer parameters.'''

        # 1 - Add all the info that will be saved in checkpoint

        model_results = {
            'best_model_train_loss' : self.best_model_train_loss,
            'best_model_training_eval_metric' : self.best_model_training_eval_metric,
            'best_model_validation_eval_metric' : self.best_model_validation_eval_metric,
        }

        training_variables = {
            'epoch': self.epoch,
            'batch_number' : self.batch_number,
            'step' : self.step,
            'validations_without_improvement' : self.validations_without_improvement,
            'validations_without_improvement_or_opt_update' : self.validations_without_improvement_or_opt_update,
            'train_loss' : self.train_loss,
            'training_eval_metric' : self.training_eval_metric,
            'validation_eval_metric' : self.validation_eval_metric,
            'best_train_loss' : self.best_train_loss,
            'best_model_train_loss' : self.best_model_train_loss,
            'best_model_training_eval_metric' : self.best_model_training_eval_metric,
            'best_model_validation_eval_metric' : self.best_model_validation_eval_metric,
            'total_trainable_params' : self.total_trainable_params,
        }

        # Extract the state dict properly (unwrap DDP if necessary)
        if self.is_distributed:
            model_state_dict = self.net.module.state_dict()
        else:
            model_state_dict = self.net.state_dict()

        checkpoint = {
            'model': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'settings': self.params,
            'model_results' : model_results,
            'training_variables' : training_variables,
        }

        end_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H:%M:%S')
        checkpoint['start_datetime'] = self.start_datetime
        checkpoint['end_datetime'] = end_datetime

        # 2 - Save the checkpoint locally (only on main process in distributed training)

        if self.is_main_process:
            checkpoint_folder = os.path.join(self.params.model_output_folder, self.params.model_name)
            checkpoint_file_name = f"{self.params.model_name}.chkpt"
            checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file_name)

            # Create directory if doesn't exists
            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)

            logger.info(f"Saving training and model information in {checkpoint_path}")
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Done.")

        # Synchronize all processes before continuing (in distributed training)
        if self.is_distributed:
            dist.barrier()

        # Delete variables to free memory
        del model_results
        del training_variables
        del checkpoint

        if self.is_main_process:
            logger.info(f"Training and model information saved.")

    def eval_and_save_best_model(self):

        if self.step > 0 and self.params.eval_and_save_best_model_every > 0 \
            and self.step % self.params.eval_and_save_best_model_every == 0:

            logger.info('Evaluating and saving the new best model (if founded)...')

            # Calculate the evaluation metrics
            self.evaluate()

            # Have we found a better model? (Better in validation metric).
            if self.validation_eval_metric > self.best_model_validation_eval_metric:

                logger.info('We found a better model!')
                if (self.params.max_overfitting_allowed is None) or \
                    (self.params.max_overfitting_allowed is not None and (self.training_eval_metric - self.validation_eval_metric) <= self.params.max_overfitting_allowed):

                # Update best model evaluation metrics
                    self.best_model_train_loss = self.train_loss
                    self.best_model_training_eval_metric = self.training_eval_metric
                    self.best_model_validation_eval_metric = self.validation_eval_metric

                    logger.info(f"Best model train loss: {self.best_model_train_loss:.3f}")
                    logger.info(f"Best model train evaluation metric: {self.best_model_training_eval_metric:.3f}")
                    logger.info(f"Best model validation evaluation metric: {self.best_model_validation_eval_metric:.3f}")

                    self.save_model()

                    # Since we found and improvement, validations_without_improvement and validations_without_improvement_or_opt_update are reseted.
                    self.validations_without_improvement = 0
                    self.validations_without_improvement_or_opt_update = 0

            else:
                # In this case the search didn't improved the model
                # We are one validation closer to do early stopping
                self.validations_without_improvement = self.validations_without_improvement + 1
                self.validations_without_improvement_or_opt_update = self.validations_without_improvement_or_opt_update + 1


            logger.info(f"Consecutive validations without improvement: {self.validations_without_improvement}")
            logger.info(f"Consecutive validations without improvement or optimizer update: {self.validations_without_improvement_or_opt_update}")
            logger.info('Evaluating and saving done.')


    def check_update_optimizer(self):

        # Update optimizer if neccesary
        if self.validations_without_improvement > 0 and self.validations_without_improvement_or_opt_update > 0\
            and self.params.update_optimizer_every > 0 \
            and self.validations_without_improvement_or_opt_update % self.params.update_optimizer_every == 0:

            if self.params.optimizer == 'sgd' or self.params.optimizer == 'adam' or self.params.optimizer == 'adamw':

                logger.info(f"Updating optimizer...")

                for param_group in self.optimizer.param_groups:

                    param_group['lr'] = param_group['lr'] * self.params.learning_rate_multiplier

                    logger.info(f"New learning rate: {param_group['lr']}")

                logger.info(f"Optimizer updated.")

            # We reset validations_without_improvement_or_opt_update since we updated the optimizer
            self.validations_without_improvement_or_opt_update = 0

        # Calculate actual learning rate
        # HACK only taking one param group lr as the overall lr (our case has only one param group)
        for param_group in self.optimizer.param_groups:
            self.learning_rate = param_group['lr']


    def check_early_stopping(self):

        if self.params.early_stopping > 0 \
            and self.validations_without_improvement >= self.params.early_stopping:

            self.early_stopping_flag = True
            logger.info(f"Doing early stopping after {self.validations_without_improvement} validations without improvement.")


    def check_print_training_info(self):

        if self.step > 0 and self.params.print_training_info_every > 0 \
            and self.step % self.params.print_training_info_every == 0:

            info_to_print = f"Epoch {self.epoch} of {self.params.max_epochs}, "
            info_to_print = info_to_print + f"batch {self.batch_number} of {self.total_batches}, "
            info_to_print = info_to_print + f"step {self.step}, "
            info_to_print = info_to_print + f"Loss {self.train_loss:.3f}, "
            info_to_print = info_to_print + f"Best validation score: {self.best_model_validation_eval_metric:.3f}..."

            logger.info(info_to_print)


    def train_single_epoch(self, epoch):

        logger.info(f"Epoch {epoch} of {self.params.max_epochs}...")

        # Switch torch to training mode
        self.net.train()

        for self.batch_number, batch_data in enumerate(self.training_generator):

            if self.params.text_feature_extractor != 'NoneTextExtractor':
                input, label, transcription_tokens_padded, transcription_tokens_mask = batch_data
            else:
                input, label = batch_data

            # Assign batch data to device
            if self.params.text_feature_extractor != 'NoneTextExtractor':
                transcription_tokens_padded = transcription_tokens_padded.long().to(self.device)
                transcription_tokens_mask = transcription_tokens_mask.long().to(self.device)

            input, label = input.float().to(self.device), label.long().to(self.device)

            if self.batch_number == 0: logger.info(f"input.size(): {input.size()}")

            # Calculate prediction and loss
            if self.params.text_feature_extractor != 'NoneTextExtractor':
                prediction  = self.net(
                    input_tensor = input,
                    transcription_tokens_padded = transcription_tokens_padded,
                    transcription_tokens_mask = transcription_tokens_mask,
                    )
            else:
                prediction  = self.net(input_tensor = input)

            self.loss = self.loss_function(prediction, label)
            self.train_loss = self.loss.item()

            # Compute backpropagation and update weights

            # Clears x.grad for every parameter x in the optimizer.
            # It’s important to call this before loss.backward(), otherwise you’ll accumulate the gradients from multiple passes.
            self.optimizer.zero_grad()

            # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True.
            # These are accumulated into x.grad for every parameter x.
            self.loss.backward()

            # optimizer.step updates the value of x using the gradient x.grad
            self.optimizer.step()

            # Calculate evaluation metrics and save the best model
            self.eval_and_save_best_model()

            # Update best loss
            if self.train_loss < self.best_train_loss:
                self.best_train_loss = self.train_loss

            self.check_update_optimizer()
            self.check_early_stopping()
            self.check_print_training_info()

            if self.params.use_weights_and_biases:
                try:
                    self.wandb_run.log(
                        {
                            "epoch" : self.epoch,
                            "batch_number" : self.batch_number,
                            "loss" : self.train_loss,
                            "learning_rate" : self.learning_rate,
                            "training_eval_metric" : self.training_eval_metric,
                            "validation_eval_metric" : self.validation_eval_metric,
                            'best_model_train_loss' : self.best_model_train_loss,
                            'best_model_training_eval_metric' : self.best_model_training_eval_metric,
                            'best_model_validation_eval_metric' : self.best_model_validation_eval_metric,
                        },
                        step = self.step
                        )
                except Exception as e:
                    logger.error('Failed at wandb.log: '+ str(e))

            if self.early_stopping_flag == True:
                break

            self.step = self.step + 1

        logger.info(f"-"*50)
        logger.info(f"Epoch {epoch} finished with:")
        logger.info(f"Loss {self.train_loss:.3f}")
        logger.info(f"Best model training evaluation metric: {self.best_model_training_eval_metric:.3f}")
        logger.info(f"Best model validation evaluation metric: {self.best_model_validation_eval_metric:.3f}")
        logger.info(f"-"*50)


    def train(self, starting_epoch, max_epochs):

        logger.info(f'Starting training for {max_epochs} epochs.')

        for self.epoch in range(starting_epoch, max_epochs):

            self.train_single_epoch(self.epoch)

            if self.early_stopping_flag == True:
                break

        logger.info('Training finished!')


    def delete_version_artifacts(self):

        logger.info(f'Starting to delete not latest checkpoint version artifacts...')

        # We want to keep only the latest checkpoint because of wandb memory storage limit

        api = wandb.Api()
        actual_run = api.run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")

        # We need to finish the run and let wandb upload all files
        wandb.run.finish()

        for artifact_version in actual_run.logged_artifacts():

            if 'latest' in artifact_version.aliases:
                latest_version = True
            else:
                latest_version = False

            if latest_version == False:
                logger.info(f'Deleting not latest artifact {artifact_version.name} from wandb...')
                artifact_version.delete(delete_aliases=True)
                logger.info(f'Deleted.')

        logger.info(f'All not latest artifacts deleted.')


    def save_model_artifact(self):

        # Save checkpoint as a wandb artifact

        logger.info(f'Starting to save checkpoint as wandb artifact...')

        # Define the artifact
        trained_model_artifact = wandb.Artifact(
            name = self.params.model_name,
            type = "trained_model",
            description = self.params.model_architecture_name,
            metadata = self.wandb_config,
        )

        # Add folder directory
        checkpoint_folder = os.path.join(self.params.model_output_folder, self.params.model_name)
        logger.info(f'checkpoint_folder {checkpoint_folder}')
        trained_model_artifact.add_dir(
            local_path = checkpoint_folder,
            skip_cache = True,
        )

        # Log the artifact
        wandb.run.log_artifact(
            trained_model_artifact
            )

        logger.info(f'Artifact saved.')


    def main(self):

        self.train(self.starting_epoch, self.params.max_epochs)
        if self.params.use_weights_and_biases and self.wandb_run.settings.mode == "online": self.save_model_artifact()
        if self.params.use_weights_and_biases and self.wandb_run.settings.mode == "online": self.delete_version_artifacts()
        if self.params.use_weights_and_biases: wandb.finish()

#----------------------------------------------------------------------


class ArgsParser:

    def __init__(self):

        self.initialize_parser()


    def initialize_parser(self):

        self.parser = argparse.ArgumentParser(
            description = 'Train a Speech Emotion Recognition model.',
            )


    def add_parser_args(self):

        #region Directory parameters
        self.parser.add_argument(
            '--train_labels_path',
            type = str,
            default = TRAIN_DEFAULT_SETTINGS['train_labels_path'],
            help = 'Path of the file containing the training examples paths and labels.',
            )

        self.parser.add_argument(
            '--train_data_dir',
            type = str,
            help = 'Optional additional directory to prepend to the train_labels_path paths.',
            )

        self.parser.add_argument(
            '--validation_labels_path',
            type = str,
            default = TRAIN_DEFAULT_SETTINGS['validation_labels_path'],
            help = 'Path of the file containing the validation examples paths and labels.',
            )

        self.parser.add_argument(
            '--validation_data_dir',
            type = str,
            help = 'Optional additional directory to prepend to the validation_labels_path paths.',
            )

        self.parser.add_argument(
            '--dataset_transcriptions_dir',
            type = str,
            default = TRAIN_DEFAULT_SETTINGS['dataset_transcriptions_path'],
            help = 'Path of the folder containing the dataset transcriptions. \
                This folder must contain one txt file per audio, named with the same as the audio.\
                For example, if the audio path is some_dir/audio_name.wav, then the dataset_transcriptions_dir \
                folder must contain a audio_name.txt file with the corresponding transcription',
            )

        self.parser.add_argument(
            '--augmentation_noises_labels_path',
            type = str,
            help = 'Path of the file containing the background noises audio paths and labels.'
            )

        self.parser.add_argument(
            '--augmentation_noises_directory',
            type = str,
            help = 'Optional additional directory to prepend to the augmentation_labels_path paths.',
            )

        self.parser.add_argument(
            '--augmentation_rirs_labels_path',
            type = str,
            help = 'Path of the file containing the RIRs audio paths.'
            )

        self.parser.add_argument(
            '--augmentation_rirs_directory',
            type = str,
            help = 'Optional additional directory to prepend to the rirs_labels_path paths.',
            )

        self.parser.add_argument(
            '--model_output_folder',
            type = str,
            default = TRAIN_DEFAULT_SETTINGS['model_output_folder'],
            help = 'Directory where model outputs and configs are saved.',
            )

        self.parser.add_argument(
            '--checkpoint_file_folder',
            type = str,
            help = 'Name of folder that contain the model checkpoint file. Mandatory if load_checkpoint is True.',
            )

        self.parser.add_argument(
            '--checkpoint_file_name',
            type = str,
            help = 'Name of the model checkpoint file. Mandatory if load_checkpoint is True.',
            )

        self.parser.add_argument(
            '--log_file_folder',
            type = str,
            default = TRAIN_DEFAULT_SETTINGS['log_file_folder'],
            help = 'Name of folder that will contain the log file.',
            )

        self.parser.add_argument(
            '--wandb_dir',
            type = str,
            help = 'An absolute path to the directory where Weight & Biases metadata and downloaded files will be stored, \
             when use_weights_and_biases is True. If not specified, this defaults to the ./wandb directory.',
            )
        #endregion

        #region Data Parameters
        self.parser.add_argument(
            '--window_secs',
            type = float,
            default = TRAIN_DEFAULT_SETTINGS['window_secs'],
            help="The number of seconds of the window"
        )
        self.parser.add_argument(
            '--stride_secs',
            type = float,
            default = TRAIN_DEFAULT_SETTINGS['stride_secs'],
            help="The number of seconds that the window will stride of the window"
        )
        self.parser.add_argument(
            '--num_folds',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS['num_folds'],
            help="The number of folds in CrossValidation"
        )
        self.parser.add_argument(
            '--random_seed',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS['random_seed'],
            help="Random seed used for patient-level fold assignment."
        )


        self.parser.add_argument(
            '--sample_rate',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS['sample_rate'],
            help = "Sample rate that you want to use (every audio loaded is resampled to this frequency)."
            )

        self.parser.add_argument(
            '--training_random_crop_secs',
            type = float,
            default = TRAIN_DEFAULT_SETTINGS['training_random_crop_secs'],
            help = 'Cut the training input audio with random_crop_secs length at a random starting point. \
                If 0, the full audio is loaded.'
            )

        self.parser.add_argument(
            '--evaluation_random_crop_secs',
            type = float,
            default = TRAIN_DEFAULT_SETTINGS['evaluation_random_crop_secs'],
            help = 'Cut the evaluation input audio with random_crop_secs length at a random starting point. \
                If 0, the full audio is loaded.'
            )

        self.parser.add_argument(
            '--num_workers',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS['num_workers'],
            help = 'num_workers to be used by the data loader.'
            )

        self.parser.add_argument(
            '--padding_type',
            type = str,
            choices = ["zero_pad", "repetition_pad"],
            help = 'Type of padding to apply to the audios. \
                zero_pad does zero left padding, repetition_pad repeats the audio.'
            )
        #endregion

        #region Data Augmentation arguments

        self.parser.add_argument(
            '--training_augmentation_prob',
            type = float,
            default = TRAIN_DEFAULT_SETTINGS['training_augmentation_prob'],
            help = 'Probability of applying data augmentation to each file. Set to 0 if not augmentation is desired.'
            )

        self.parser.add_argument(
            '--evaluation_augmentation_prob',
            type = float,
            default = TRAIN_DEFAULT_SETTINGS['evaluation_augmentation_prob'],
            help = 'Probability of applying data augmentation to each file. Set to 0 if not augmentation is desired.'
            )

        self.parser.add_argument(
            '--augmentation_window_size_secs',
            type = float,
            default = TRAIN_DEFAULT_SETTINGS['augmentation_window_size_secs'],
            help = 'Cut the audio with augmentation_window_size_secs length at a random starting point. \
                If 0, the full audio is loaded.'
            )

        self.parser.add_argument(
            '--augmentation_effects',
            type = str,
            nargs = '+',
            choices = ["apply_speed_perturbation", "apply_reverb", "add_background_noise"],
            help = 'Effects to augment the data. One or many can be choosen.'
            )
        #endregion

        #region Network Parameters

        self.parser.add_argument(
            '--speech_feature_extractor',
            type = str,
            default = TRAIN_DEFAULT_SETTINGS['speech_feature_extractor'],
            choices = ['WAVLM_BASE', 'WAVLM_BASE_PLUS', 'WAVLM_LARGE', 'WAV2VEC2_LARGE_LV60K', 'WAV2VEC2_XLSR_300M', 'WAV2VEC2_XLSR_1B', 'HUBERT_LARGE'],
            help = 'Type of extractor used to generate features from speech. \
                It will take an audio waveform and output a sequence of vectors (features).'
            )

        self.parser.add_argument(
            '--speech_feature_extractor_output_vectors_dimension',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS['speech_feature_extractor_output_vectors_dimension'],
            help = 'Dimension of each vector that will be the output of the speech feature extractor.'
            )

        self.parser.add_argument(
            '--text_feature_extractor',
            type = str,
            default = TRAIN_DEFAULT_SETTINGS['text_feature_extractor'],
            choices = ['BERT_BASE_UNCASED', 'BERT_BASE_CASED', 'BERT_LARGE_UNCASED', 'BERT_LARGE_CASED', 'ROBERTA_LARGE', 'MODERN_BERT_BASE', 'MODERN_BERT_LARGE'],
            help = 'Type of extractor used to generate features from text. \
                It will take text and output a sequence of vectors (features).'
            )

        self.parser.add_argument(
            '--text_feature_extractor_output_vectors_dimension',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS['text_feature_extractor_output_vectors_dimension'],
            help = 'Dimension of each vector that will be the output of the text feature extractor.'
            )

        self.parser.add_argument(
            '--speech_adapter',
            type = str,
            default = TRAIN_DEFAULT_SETTINGS['speech_adapter'],
            choices = ['NoneAdapter', 'LinearAdapter', 'NonLinearAdapter'],
            help = 'Type of adapter used to project speech features.'
            )

        self.parser.add_argument(
            '--speech_adapter_output_vectors_dimension',
            type = int,
            help = 'Dimension of each vector that will be the output of the speech adapter layer.',
            )

        self.parser.add_argument(
            '--text_adapter',
            type = str,
            default = TRAIN_DEFAULT_SETTINGS['text_adapter'],
            choices = ['NoneAdapter', 'LinearAdapter', 'NonLinearAdapter'],
            help = 'Type of adapter used to project text features.'
            )

        self.parser.add_argument(
            '--text_adapter_output_vectors_dimension',
            type = int,
            help = 'Dimension of each vector that will be the output of the text adapter layer.',
            )

        self.parser.add_argument(
            '--seq_to_seq_method',
            type = str,
            default = TRAIN_DEFAULT_SETTINGS['seq_to_seq_method'],
            choices = ['NoneSeqToSeq', 'SelfAttention', 'MultiHeadAttention', 'TransformerStacked', 'ReducedMultiHeadAttention', 'CrossAttention', 'CrossAttentionReduced'],
            help = 'Sequence to sequence component after the linear projection layer of the model.',
            )

        self.parser.add_argument(
            '--seq_to_seq_heads_number',
            type = int,
            help = 'Number of heads for the seq_to_seq layer of the pooling component \
                (only for MHA based seq_to_seq options).',
            )

        self.parser.add_argument(
            '--skip_connections',
            action = argparse.BooleanOptionalAction,
            default = TRAIN_DEFAULT_SETTINGS['skip_connections'],
            help="Introduces skip connection before seq_to_seq part"
        )
        self.parser.add_argument(
            '--transformer_n_blocks',
            type = int,
            help = 'Number of transformer blocks to stack in the seq_to_seq component of the pooling. \
                (Only for seq_to_seq_method = TransformerStacked).',
            )

        self.parser.add_argument(
            '--transformer_expansion_coef',
            type = int,
            help = "Number you want to multiply by the size of the hidden layer of the transformer block's feed forward net. \
                (Only for seq_to_seq_method = TransformerBlock)."
            )

        self.parser.add_argument(
            '--transformer_drop_out',
            type = float,
            help = 'Dropout probability to use in the feed forward component of the transformer block.\
                (Only for seq_to_seq_method = TransformerBlock).'
            )

        self.parser.add_argument(
            '--seq_to_one_method',
            type = str,
            default = TRAIN_DEFAULT_SETTINGS['seq_to_one_method'],
            choices = ['StatisticalPooling', 'AttentionPooling'],
            help = 'Type of pooling method applied to the output sequence to sequence component of the model.',
            )

        self.parser.add_argument(
            '--seq_to_seq_input_dropout',
            type = float,
            default = TRAIN_DEFAULT_SETTINGS['seq_to_seq_input_dropout'],
            help = 'Dropout probability to use in the seq to seq component input.'
            )

        self.parser.add_argument(
            '--seq_to_one_input_dropout',
            type = float,
            default = TRAIN_DEFAULT_SETTINGS['seq_to_one_input_dropout'],
            help = 'Dropout probability to use in the seq to one component input.'
            )

        self.parser.add_argument(
            '--classifier_layer_drop_out',
            type = float,
            default = TRAIN_DEFAULT_SETTINGS['classifier_layer_drop_out'],
            help = 'Dropout probability to use in the classfifer component.'
            )

        self.parser.add_argument(
            '--classifier_hidden_layers',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS['classifier_hidden_layers'],
            help = 'Number of hidden layers in the classifier layer.',
            )

        self.parser.add_argument(
            '--classifier_hidden_layers_width',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS['classifier_hidden_layers_width'],
            help = 'Width of every hidden layer in the classifier layer.',
            )

        self.parser.add_argument(
            '--number_classes',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS['number_classes'],
            help = "Number of classes to classify.",
            )
        #endregion

        #region Training Parameters

        self.parser.add_argument(
            '--max_epochs',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS['max_epochs'],
            help = 'Max number of epochs to train.',
            )

        self.parser.add_argument(
            '--max_overfitting_allowed',
            type = float,
            help = "When validating the model is not saved if \
                (best_model_training_eval_metric - best_model_validation_eval_metric) > max_overfitting_allowed.",
            )

        self.parser.add_argument(
            '--training_batch_size',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS['training_batch_size'],
            help = "Size of training batches.",
            )

        self.parser.add_argument(
            '--evaluation_batch_size',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS['evaluation_batch_size'],
            help = "Size of evaluation batches.",
            )

        self.parser.add_argument(
            '--eval_and_save_best_model_every',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS['eval_and_save_best_model_every'],
            help = "The model is evaluated on train and validation sets and saved every eval_and_save_best_model_every steps. \
                Set to 0 if you don't want to execute this utility.",
            )

        self.parser.add_argument(
            '--print_training_info_every',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS['print_training_info_every'],
            help = "Training info is printed every print_training_info_every steps. \
                Set to 0 if you don't want to execute this utility.",
            )

        self.parser.add_argument(
            '--early_stopping',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS['early_stopping'],
            help = "Training is stopped if there are early_stopping consectuive validations without improvement. \
                Set to 0 if you don't want to execute this utility.",
            )

        self.parser.add_argument(
            '--load_checkpoint',
            action = argparse.BooleanOptionalAction,
            default = TRAIN_DEFAULT_SETTINGS['load_checkpoint'],
            help = 'Set to True if you want to load a previous checkpoint and continue training from that point. \
                Loaded parameters will overwrite all inputted parameters.',
            )
        #endregion

        #region Optimization arguments

        self.parser.add_argument(
            '--optimizer',
            type = str,
            choices = ['adam', 'sgd', 'rmsprop', 'adamw'],
            default = TRAIN_DEFAULT_SETTINGS['optimizer'],
            )

        self.parser.add_argument(
            '--learning_rate',
            type = float,
            default = TRAIN_DEFAULT_SETTINGS['learning_rate'],
            )

        self.parser.add_argument(
            '--learning_rate_multiplier',
            type = float,
            default = TRAIN_DEFAULT_SETTINGS['learning_rate_multiplier'],
            )

        self.parser.add_argument(
            '--weight_decay',
            type = float,
            default = TRAIN_DEFAULT_SETTINGS['weight_decay'],
            )

        self.parser.add_argument(
            '--update_optimizer_every',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS['update_optimizer_every'],
            help = "Some optimizer parameters will be updated every update_optimizer_every consecutive validations without improvement. \
                Set to 0 if you don't want to execute this utility.",
            )

        self.parser.add_argument(
            '--loss',
            type = str,
            choices = ['CrossEntropy', 'FocalLoss'],
            default = TRAIN_DEFAULT_SETTINGS['loss'],
            )

        self.parser.add_argument(
            "--weighted_loss",
            action = argparse.BooleanOptionalAction,
            default = TRAIN_DEFAULT_SETTINGS['weighted_loss'],
            help = "Set the weight parameter of the loss to a tensor representing the inverse frequency of each class.",
            )

        #endregion

        #region Verbosity and debug Parameters

        self.parser.add_argument(
            "--use_weights_and_biases",
            action = argparse.BooleanOptionalAction,
            default = TRAIN_DEFAULT_SETTINGS['use_weights_and_biases'],
            help = "Use weights and Biases.",
            )

        #endregion


        #region whisper
        self.parser.add_argument(
            "--whisper_model_name",
            type = str,
            default = TRAIN_DEFAULT_SETTINGS['whisper_model_name'],
            help="The name of whisper model"
        )
        self.parser.add_argument(
            "--whisper_language",
            type = str,
            default = TRAIN_DEFAULT_SETTINGS['whisper_language'],
            help="The language parsed for whisper to transcribe"
        )
        self.parser.add_argument(
            "--transcription_cache_dir",
            type = str,
            default = TRAIN_DEFAULT_SETTINGS['transcription_cache_dir'],
            help="Optional directory for cached segment transcriptions."
        )
        #endregion


    def main(self):

        self.add_parser_args()
        self.arguments = self.parser.parse_args()

# # ---------------------------------------------------------------------

if __name__ == "__main__":

    args_parser = ArgsParser()
    args_parser.main()
    trainer_parameters = args_parser.arguments

    trainer = Trainer(trainer_parameters)
    trainer.main()