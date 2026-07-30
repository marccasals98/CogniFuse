"""Data loading utilities for PPA subtype classification.

This module is designed for the precomputed-feature workflow used by the
WAB/PPA project. It:

* read the project CSV directly;
* keep patient-level train/validation separation;
* split every recording into fixed, overlapping windows;
* treat every window as an independent dataset item;
* load the raw waveform for that window;
* transcribe that same window with Whisper and cache the transcription;
* return ``waveform, label, transcription_tokens``.

Typical use::

    train_dataset = ADDataset(
        csv_path="/path/to/labels.csv",
        audio_dir="/path/to/WAB_samples",
        input_parameters=parameters,
        window_secs=14,
        stride_secs=7,
        split="train",
        fold=0,
    )

    val_dataset = ADDataset(
        csv_path="/path/to/labels.csv",
        audio_dir="/path/to/WAB_samples",
        input_parameters=parameters,
        window_secs=14,
        stride_secs=7,
        split="val",
        fold=0,
        augmentation_prob=0,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
"""


from __future__ import annotations

from torch.utils import data
from torch.nn.utils.rnn import pad_sequence

import copy
import logging
import os
import random
from typing import Dict, List, Optional, Sequence

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio

try:
    import whisper
except ImportError:
    whisper = None

try:
    from augmentation import DataAugmentator
except ImportError:
    DataAugmentator = None


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

class ADDataset(data.Dataset):
    """PPA dataset based on overlapping raw-audio windows.

    Each dataset item corresponds to one fixed-duration segment, not to one
    complete recording. For example, with ``window_secs=14`` and
    ``stride_secs=7`` a recording is represented by windows beginning at
    0, 7, 14, 21, ... seconds.

    Expected CSV columns for the current project:

    * first unnamed column: audio filename;
    * ``NHC ID HSP``: patient identifier;
    * ``DX_Pilar``: diagnosis.

    Parameters
    ----------
    csv_path:
        Path to ``labels.csv``.
    audio_dir:
        Directory containing the .alac, .wav and .mp3 recordings.
    input_parameters:
        Project parameter object. It must at least provide ``sample_rate`` and
        ``text_feature_extractor``.
    window_secs:
        Duration of every segment in seconds.
    stride_secs:
        Distance in seconds between consecutive segment starts.
    split:
        ``"train"`` or ``"val"``/``"validation"``/``"test"``.
    fold:
        Validation fold index. When ``None``, folds 0..79% are used for train
        and the remaining folds for validation, preserving the previous 80/20
        behaviour.
    num_folds:
        Number of patient-level folds.
    target_classes:
        Diagnoses included in the task. Defaults to the three PPA subtypes.
    ignore_labels:
        Extra diagnosis values to exclude.
    augmentation_prob:
        Probability of augmenting a returned training waveform.
    whisper_model_name:
        Whisper model used when a cached segment transcription is unavailable.
    whisper_language:
        Language passed to Whisper. Defaults to Spanish.
    transcription_cache_dir:
        Optional explicit cache directory. By default it is created inside
        ``audio_dir`` using the window and stride values.
    """


    def __init__(
        self,
        input_parameters,
        split="train",
        fold=None,
        num_folds=5,
        target_classes=None,
        ignore_labels=None,
    ):
        self.parameters = copy.deepcopy(input_parameters)

        self.csv_path = self.parameters.csv_path
        self.audio_dir = self.parameters.audio_dir
        self.window_secs = float(self.parameters.window_secs)
        self.stride_secs = float(self.parameters.stride_secs)
        self.split = split.lower()
        self.fold = fold
        self.num_folds = int(self.parameters.num_folds)
        self.augmentation_prob = float(self.parameters.augmentation_prob)
        self.whisper_model_name = self.parameters.whisper_model_name
        self.whisper_language = self.parameters.whisper_language
        self.random_seed = int(self.parameters.random_seed)

        # Transform the window from seconds to number of samples:
        self.window_samples = int(round(self.window_secs * self.parameters.sample_rate))

        if self.window_secs <= 0:
            raise ValueError("window_secs must be greater than zero.")
        if self.stride_secs <= 0:
            raise ValueError("stride_secs must be greater than zero.")
        if self.num_folds < 2:
            raise ValueError("num_folds must be at least 2.")
        if self.split not in {"train", "val", "validation", "test"}:
            raise ValueError(
                "split must be 'train', 'val', 'validation', or 'test'."
            )
        if target_classes is None:
            target_classes = ["lvPPA", "nfPPA", "svPPA"]
        if ignore_labels is None:
            ignore_labels = []

        self.target_classes = [
            label for label in target_classes if label not in ignore_labels
        ]
        # maps each class to a class id.
        self.label_map = {
            label: class_id
            for class_id, label in enumerate(self.target_classes)
        }
        self.label_names = list(self.target_classes)
        self.num_classes = len(self.label_map)
        if self.num_classes == 0:
            raise ValueError("No target classes remain after applying ignore_labels.")

        # Folder where dataset stores and reuses Whisper transcriptions.
        if transcription_cache_dir is None:
            transcription_cache_dir = os.path.join(
                self.audio_dir,
                f"trans_w{self._format_number(self.window_secs)}_"
                f"s{self.stride_secs:.2f}",
            )
        self.transcription_cache_dir = transcription_cache_dir
        os.makedirs(self.transcription_cache_dir, exist_ok=True)

        if self.augmentation_prob > 0:
            self.init_data_augmentator()

        self.init_text_feature_extractor_tokenizer()
        self.init_whisper_model()

        labels_df = self.load_and_prepare_csv()
        self.df = self.create_patient_split(labels_df)
        self.segments = self.build_segments()

        self.num_files = len(self.segments)
        self.class_counts = self.compute_class_counts()

        logger.info("Label map: %s", self.label_map)
        logger.info(
            "[%s] Loaded %d recordings and created %d segments: %s",
            self.split,
            len(self.df),
            self.num_files,
            self.class_counts,
        )

    
    def get_classes_weights(self):
        """
        Compute a tensor representing the inverse frequency of each class.
        This will be used (if so) to set the weight parameter of the loss.
        """

        dataset_labels = [path.strip().split('\t')[1] for path in self.labels_lines]

        weights_series = pd.Series(dataset_labels).value_counts(normalize = True, dropna = False)
        weights_df = pd.DataFrame(weights_series).reset_index()
        weights_df.columns = ["class_id", "weight"]
        weights_df["weight"] = 1 / weights_df["weight"]
        weights_df = weights_df.sort_values("class_id", ascending=True)

        weights = weights_df["weight"].to_list()

        for class_id in range(len(weights)):
            logger.info(f"Class_id {class_id} weight: {weights[class_id]}")

        return weights


    def init_data_augmentator(self):

        self.data_augmentator = DataAugmentator(
            self.parameters.augmentation_noises_directory,
            self.parameters.augmentation_noises_labels_path,
            self.parameters.augmentation_rirs_directory,
            self.parameters.augmentation_rirs_labels_path,
            self.parameters.augmentation_window_size_secs,
            self.parameters.augmentation_effects,
        )


    def init_text_feature_extractor_tokenizer(self):

        if self.parameters.text_feature_extractor == "BERT_BASE_UNCASED":
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
        elif self.parameters.text_feature_extractor == "BERT_BASE_CASED":
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
        elif self.parameters.text_feature_extractor == "BERT_LARGE_UNCASED":
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-large-uncased')
        elif self.parameters.text_feature_extractor == "BERT_LARGE_CASED":
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-large-cased')
        elif self.parameters.text_feature_extractor == "ROBERTA_LARGE":
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'roberta-large')
        elif self.parameters.text_feature_extractor == "MODERN_BERT_BASE":
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'answerdotai/ModernBERT-base')
        elif self.parameters.text_feature_extractor == "MODERN_BERT_LARGE":
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'answerdotai/ModernBERT-large', reference_compile=False)
        else:
            raise Exception('No text_feature_extractor choice found.')


    def pad_waveform(self, waveform, padding_type, random_crop_samples):

        if padding_type == "zero_pad":
            pad_left = max(0, self.random_crop_samples - waveform.shape[-1])
            padded_waveform = torch.nn.functional.pad(waveform, (pad_left, 0), mode = "constant")
        elif padding_type == "repetition_pad":
            necessary_repetitions = int(np.ceil(random_crop_samples / waveform.size(-1)))
            if waveform.dim() == 1:
                padded_waveform = waveform.repeat(necessary_repetitions)
            else:
                padded_waveform = waveform.repeat(1, necessary_repetitions)
        else:
            raise Exception('No padding choice found.')

        return padded_waveform


    def sample_audio_window(self, waveform, random_crop_samples):

        waveform_total_samples = waveform.size()[-1]

        assert random_crop_samples <= waveform_total_samples, f"random_crop_samples ({random_crop_samples}) must be less than waveform_total_samples ({waveform_total_samples})!"

        random_start_index = randint(0, waveform_total_samples - random_crop_samples)
        end_index = random_start_index + random_crop_samples

        cropped_waveform =  waveform[random_start_index : end_index]

        return cropped_waveform


    def normalize(self, waveform):

        if self.waveforms_mean is not None and self.waveforms_std is not None:
            normalized_waveform = (waveform - self.waveforms_mean) / (self.waveforms_std + 0.000001)
        else:
            normalized_waveform = waveform

        return normalized_waveform


    def process_waveform(self, waveform, original_sample_rate):

        if original_sample_rate != self.parameters.sample_rate:
            logger.warning(f"resampling from {original_sample_rate} to {self.parameters.sample_rate}")
            waveform = torchaudio.functional.resample(
                waveform = waveform,
                orig_freq = original_sample_rate,
                new_freq = self.parameters.sample_rate,
                )

        # randomly choose to do augmentation, according to self.augmentation_prob
        if random.uniform(0, 0.999) > 1 - self.augmentation_prob:
            waveform = self.data_augmentator(waveform, self.parameters.sample_rate)

        # we use squeeze to get ride of channels, that should be mono
        waveform = waveform.squeeze(0)
        if waveform.dim() > 1:
            waveform = torch.mean(waveform, dim=0)

        if self.random_crop_secs > 0:
            # We make padding to allow cropping longer segments
            # (If not, we can only crop at most the duration of the shortest audio)
            if self.random_crop_samples > waveform.size(-1):
                waveform = self.pad_waveform(waveform, self.parameters.padding_type, self.random_crop_samples)

            # TODO torchaudio.load has frame_offset and num_frames params. Providing num_frames and frame_offset arguments is more efficient
            waveform = self.sample_audio_window(
                waveform,
                random_crop_samples = self.random_crop_samples,
                )
        else:
            # HACK don't understand why, I have to do this slicing (which sample_audio_window does) to make dataloader work
            waveform =  waveform[:]

        # Delete this, each speech feature extractor should do the corresponding normalization
        #waveform = self.normalize(waveform)
        return waveform


    def get_transcription(self, audio_path):

        if audio_path.endswith(".wav"):
            file_name = audio_path.split("/")[-1].replace(".wav", ".txt")
        elif audio_path.endswith(".mp3"):
            file_name = audio_path.split("/")[-1].replace(".mp3", ".txt")
        transcription_path = os.path.join(self.parameters.dataset_transcriptions_dir, file_name)

        with open(transcription_path, 'r') as transcription_file:
            transcription = transcription_file.readlines()

        if len(transcription) != 1:
            raise Exception(f"Problems with the following transcription: {audio_path}")
        else:
            transcription = transcription[0]

        return transcription


    def get_transcription_tokens(self, transcription):

        indexed_tokens = self.tokenizer.encode(transcription, add_special_tokens=True)
        #tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = torch.tensor(indexed_tokens)

        return tokens_tensor


    def __getitem__(self, index):
        """
        Generates one sample of data (mandatory torch method).
        """

        # Each labels_line is like: audio_path\tlabel
        label_tuple = self.labels_lines[index].strip().split('\t')

        audio_path = label_tuple[0]
        label = label_tuple[1]

        # We transcribe before processing the waveform
        transcription = self.get_transcription(audio_path)
        transcription_tokens = self.get_transcription_tokens(transcription)
        if len (transcription_tokens) > self.tokenizer.model_max_length:
            logger.info(f"Transcription tokens length: {len(transcription_tokens)}, the transcription will be cut up to {self.tokenizer.model_max_length} tokens.")
            transcription_tokens = transcription_tokens[:self.tokenizer.model_max_length]
        # By default, the resulting tensor object has dtype=torch.float32 and its value range is normalized within [-1.0, 1.0]!
        waveform, original_sample_rate = torchaudio.load(audio_path)
        waveform = self.process_waveform(waveform, original_sample_rate)

        labels = np.array(int(label))

        return waveform, labels, transcription_tokens


    def __len__(self):
        # Mandatory torch method
        return self.num_files
