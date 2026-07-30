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

        transcription_cache_dir = self.parameters.transcription_cache_dir

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

    # -----------------------------------------------------------------
    # Dataset preparation
    # -----------------------------------------------------------------

    @staticmethod
    def _format_number(value):
        """Format integer-valued seconds without a trailing decimal."""
        return str(int(value)) if float(value).is_integer() else str(value)

    @staticmethod
    def _find_column(df, preferred_names, default=None):
        """Find a CSV column using exact or case-insensitive matching."""
        for name in preferred_names:
            if name in df.columns:
                return name

        normalized_columns = {
            str(column).strip().lower(): column for column in df.columns
        }
        for name in preferred_names:
            normalized_name = str(name).strip().lower()
            if normalized_name in normalized_columns:
                return normalized_columns[normalized_name]

        if default is not None:
            return default

        raise KeyError(
            f"Could not find any of {preferred_names}. "
            f"Available columns: {list(df.columns)}"
        )

    def load_and_prepare_csv(self):
        """Read labels.csv, standardize columns, and keep target diagnoses."""
        labels_df = pd.read_csv(self.csv_path)

        if labels_df.empty:
            raise ValueError(f"The labels CSV is empty: {self.csv_path}")

        filename_column = self._find_column(
            labels_df,
            ["filename", "audio_path", "audio", "file"],
            default=labels_df.columns[0],
        )
        patient_column = self._find_column(
            labels_df,
            ["NHC ID HSP", "nhc_id_hsp", "patient_id"],
        )
        label_column = self._find_column(
            labels_df,
            ["DX_Pilar", "dx_pilar", "diagnosis", "label"],
        )

        labels_df = labels_df.rename(
            columns={
                filename_column: "filename",
                patient_column: "patient_id",
                label_column: "diagnosis",
            }
        )

        labels_df["filename"] = labels_df["filename"].astype(str).str.strip()
        labels_df["patient_id"] = labels_df["patient_id"].astype(str).str.strip()
        labels_df["diagnosis"] = labels_df["diagnosis"].astype(str).str.strip()

        labels_df = labels_df[
            labels_df["diagnosis"].isin(self.label_map.keys())
            & labels_df["filename"].ne("")
            & labels_df["filename"].ne("nan")
            & labels_df["patient_id"].ne("")
            & labels_df["patient_id"].ne("nan")
        ].reset_index(drop=True)

        if labels_df.empty:
            raise ValueError(
                "No valid rows remain after filtering the CSV to "
                f"{self.target_classes}."
            )

        # Detect patients with contradictory labels before splitting.
        labels_per_patient = labels_df.groupby("patient_id")["diagnosis"].nunique()
        contradictory_patients = labels_per_patient[labels_per_patient > 1]
        if not contradictory_patients.empty:
            raise ValueError(
                "Some patients have more than one diagnosis in the CSV: "
                f"{contradictory_patients.index.tolist()[:10]}"
            )

        return labels_df

    def create_patient_split(self, labels_df):
        """Create a deterministic, approximately stratified patient split."""
        patient_labels = labels_df.groupby("patient_id")["diagnosis"].first()

        patient_to_fold = {}
        rng = np.random.default_rng(self.random_seed)

        for diagnosis in self.target_classes:
            class_patients = patient_labels[
                patient_labels == diagnosis
            ].index.to_numpy(copy=True)

            rng.shuffle(class_patients)

            for index, patient_id in enumerate(class_patients):
                patient_to_fold[patient_id] = index % self.num_folds

        if self.fold is not None:
            if not 0 <= int(self.fold) < self.num_folds:
                raise ValueError(
                    f"fold must be between 0 and {self.num_folds - 1}."
                )

            validation_fold = int(self.fold)

            if self.split == "train":
                selected_patients = {
                    patient_id
                    for patient_id, patient_fold in patient_to_fold.items()
                    if patient_fold != validation_fold
                }
            else:
                selected_patients = {
                    patient_id
                    for patient_id, patient_fold in patient_to_fold.items()
                    if patient_fold == validation_fold
                }
        else:
            # Backward-compatible deterministic 80/20 split based on folds.
            train_fold_limit = max(1, int(self.num_folds * 0.8))

            train_patients = {
                patient_id
                for patient_id, patient_fold in patient_to_fold.items()
                if patient_fold < train_fold_limit
            }
            validation_patients = set(patient_to_fold) - train_patients

            selected_patients = (
                train_patients if self.split == "train" else validation_patients
            )

        split_df = labels_df[
            labels_df["patient_id"].isin(selected_patients)
        ].reset_index(drop=True)

        if split_df.empty:
            logger.warning(
                "The %s split contains no recordings. This can happen when a "
                "class has fewer patients than num_folds.",
                self.split,
            )

        return split_df

    def build_segments(self):
        """Precompute every complete overlapping window from every recording."""
        segments = []
        missing_files = 0
        short_files = 0

        for _, row in self.df.iterrows():
            audio_path = os.path.join(self.audio_dir, row["filename"])

            if not os.path.exists(audio_path):
                missing_files += 1
                logger.warning("Audio file not found: %s", audio_path)
                continue

            try:
                duration = self.get_audio_duration(audio_path)
            except Exception as error:
                logger.error("Could not inspect %s: %s", audio_path, error)
                continue

            if duration < self.window_secs:
                short_files += 1
                logger.warning(
                    "Skipping %.2f-second file because the window is %.2f "
                    "seconds: %s",
                    duration,
                    self.window_secs,
                    audio_path,
                )
                continue

            start_sec = 0.0

            # A small tolerance avoids losing a valid final window because of
            # floating-point representation.
            while start_sec + self.window_secs <= duration + 1e-8:
                segments.append(
                    {
                        "audio_path": audio_path,
                        "filename": row["filename"],
                        "patient_id": row["patient_id"],
                        "start_sec": start_sec,
                        "label": self.label_map[row["diagnosis"]],
                    }
                )
                start_sec += self.stride_secs

        if missing_files:
            logger.warning("Missing audio files: %d", missing_files)
        if short_files:
            logger.warning("Recordings shorter than one window: %d", short_files)

        return segments

    @staticmethod
    def get_audio_duration(audio_path):
        """Read duration without loading the complete waveform when possible."""
        try:
            info = torchaudio.info(audio_path)
            if info.sample_rate <= 0:
                raise RuntimeError("Invalid sample rate returned by torchaudio.info")
            return info.num_frames / info.sample_rate
        except (AttributeError, RuntimeError, OSError):
            return librosa.get_duration(path=audio_path)

    # -----------------------------------------------------------------
    # Class statistics
    # -----------------------------------------------------------------

    def compute_class_counts(self):
        """Return segment counts as ``{class_id: number_of_segments}``."""
        class_counts = {
            class_id: 0 for class_id in range(self.num_classes)
        }

        for segment in self.segments:
            class_counts[segment["label"]] += 1

        return class_counts

    def get_classes_weights(self):
        """Return inverse segment-frequency weights for CrossEntropyLoss."""
        total_segments = len(self.segments)
        weights = []

        for class_id in range(self.num_classes):
            class_count = self.class_counts[class_id]

            if class_count == 0 or total_segments == 0:
                weight = 0.0
                logger.warning(
                    "Class %d (%s) has no segments in the %s split.",
                    class_id,
                    self.label_names[class_id],
                    self.split,
                )
            else:
                class_frequency = class_count / total_segments
                weight = 1.0 / class_frequency

            weights.append(weight)
            logger.info(
                "Class_id %d (%s) weight: %.6f",
                class_id,
                self.label_names[class_id],
                weight,
            )

        return weights

    # -----------------------------------------------------------------
    # Augmentation and models
    # -----------------------------------------------------------------

    def init_data_augmentator(self):
        """Initialize the project's waveform augmentator."""
        if DataAugmentator is None:
            raise ImportError(
                "augmentation_prob is greater than zero, but augmentation.py "
                "could not be imported."
            )

        self.data_augmentator = DataAugmentator(
            self.parameters.augmentation_noises_directory,
            self.parameters.augmentation_noises_labels_path,
            self.parameters.augmentation_rirs_directory,
            self.parameters.augmentation_rirs_labels_path,
            self.parameters.augmentation_window_size_secs,
            self.parameters.augmentation_effects,
        )

    def init_whisper_model(self):
        """Load Whisper for cache misses.

        Cached transcriptions can still be used when Whisper is unavailable.
        An error is raised only if a requested segment has no cached text.
        """
        self.whisper_model = None

        if not self.whisper_model_name:
            logger.info("Whisper loading disabled; transcription cache only.")
            return

        if whisper is None:
            logger.warning(
                "The whisper package is unavailable. Existing cached "
                "transcriptions can be read, but cache misses will fail."
            )
            return

        try:
            logger.info("Loading Whisper model: %s", self.whisper_model_name)
            self.whisper_model = whisper.load_model(self.whisper_model_name)
        except Exception as error:
            logger.warning(
                "Whisper could not be loaded: %s. Existing cached "
                "transcriptions can still be used.",
                error,
            )

    def init_text_feature_extractor_tokenizer(self):
        """Initialize the tokenizer selected in the project parameters."""
        extractor = self.parameters.text_feature_extractor

        if extractor == "BERT_BASE_UNCASED":
            model_name = "bert-base-uncased"
        elif extractor == "BERT_BASE_CASED":
            model_name = "bert-base-cased"
        elif extractor == "BERT_LARGE_UNCASED":
            model_name = "bert-large-uncased"
        elif extractor == "BERT_LARGE_CASED":
            model_name = "bert-large-cased"
        elif extractor == "ROBERTA_LARGE":
            model_name = "roberta-large"
        elif extractor == "MODERN_BERT_BASE":
            model_name = "answerdotai/ModernBERT-base"
        elif extractor == "MODERN_BERT_LARGE":
            model_name = "answerdotai/ModernBERT-large"
        else:
            raise ValueError(
                "No valid text_feature_extractor choice was found: "
                f"{extractor}"
            )

        self.tokenizer = torch.hub.load(
            "huggingface/pytorch-transformers",
            "tokenizer",
            model_name,
        )

    # -----------------------------------------------------------------
    # Waveform processing
    # -----------------------------------------------------------------

    def load_audio_segment(self, audio_path, start_sec):
        """Load one segment, convert to mono, and resample in one operation."""
        waveform, _ = librosa.load(
            audio_path,
            sr=self.parameters.sample_rate,
            mono=True,
            offset=float(start_sec),
            duration=self.window_secs,
        )

        waveform = torch.from_numpy(waveform).float()
        waveform = self.ensure_fixed_length(waveform)

        return waveform

    def pad_waveform(self, waveform, target_samples):
        """Pad a short decoded segment using the configured padding strategy."""
        padding_type = getattr(self.parameters, "padding_type", "zero_pad")

        if waveform.numel() == 0:
            return torch.zeros(target_samples, dtype=torch.float32)

        if padding_type == "zero_pad":
            missing_samples = max(0, target_samples - waveform.shape[-1])
            return torch.nn.functional.pad(
                waveform,
                (0, missing_samples),
                mode="constant",
            )

        if padding_type == "repetition_pad":
            necessary_repetitions = int(
                np.ceil(target_samples / waveform.size(-1))
            )
            return waveform.repeat(necessary_repetitions)

        raise ValueError(
            f"Unknown padding_type {padding_type!r}. "
            "Use 'zero_pad' or 'repetition_pad'."
        )

    def ensure_fixed_length(self, waveform):
        """Ensure every returned waveform contains exactly one full window."""
        if waveform.size(-1) < self.window_samples:
            waveform = self.pad_waveform(waveform, self.window_samples)

        return waveform[: self.window_samples]

    def process_waveform(self, waveform):
        """Optionally augment a training segment while preserving its length."""
        should_augment = (
            self.split == "train"
            and self.augmentation_prob > 0
            and random.random() < self.augmentation_prob
        )

        if should_augment:
            waveform = self.data_augmentator(
                waveform,
                self.parameters.sample_rate,
            )

        waveform = waveform.squeeze()

        if waveform.dim() > 1:
            waveform = torch.mean(waveform, dim=0)

        return self.ensure_fixed_length(waveform)

    # -----------------------------------------------------------------
    # Transcription processing
    # -----------------------------------------------------------------

    def get_transcription_cache_path(self, audio_path, start_sec):
        """Return the text-cache path for one audio segment."""
        audio_stem = os.path.splitext(os.path.basename(audio_path))[0]
        cache_filename = f"{audio_stem}_{start_sec:.2f}.txt"
        return os.path.join(self.transcription_cache_dir, cache_filename)

    def get_transcription(self, audio_path, start_sec, waveform):
        """Read a cached transcription or transcribe the segment with Whisper."""
        cache_path = self.get_transcription_cache_path(audio_path, start_sec)

        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as transcription_file:
                return transcription_file.read().strip()

        if self.whisper_model is None:
            raise RuntimeError(
                "No cached transcription exists for this segment and Whisper "
                f"is unavailable: {cache_path}"
            )

        audio_16k = waveform

        if self.parameters.sample_rate != 16000:
            audio_16k = torchaudio.functional.resample(
                waveform=waveform,
                orig_freq=self.parameters.sample_rate,
                new_freq=16000,
            )

        result = self.whisper_model.transcribe(
            audio_16k.cpu().numpy(),
            fp16=torch.cuda.is_available(),
            language=self.whisper_language,
        )
        transcription = result.get("text", "").strip()

        temporary_path = cache_path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as transcription_file:
            transcription_file.write(transcription)
        os.replace(temporary_path, cache_path)

        return transcription

    def get_transcription_tokens(self, transcription):
        """Convert one segment transcription to token IDs."""
        indexed_tokens = self.tokenizer.encode(
            transcription,
            add_special_tokens=True,
        )
        transcription_tokens = torch.tensor(indexed_tokens, dtype=torch.long)

        if len(transcription_tokens) > self.tokenizer.model_max_length:
            logger.info(
                "Transcription has %d tokens and will be truncated to %d.",
                len(transcription_tokens),
                self.tokenizer.model_max_length,
            )
            transcription_tokens = transcription_tokens[
                : self.tokenizer.model_max_length
            ]

        return transcription_tokens

    # -----------------------------------------------------------------
    # Mandatory torch Dataset methods
    # -----------------------------------------------------------------

    def __getitem__(self, index):
        """Return one overlapping audio window and its text/label data."""
        segment = self.segments[index]

        audio_path = segment["audio_path"]
        start_sec = segment["start_sec"]
        label = segment["label"]

        # Transcribe the clean waveform. Augmentation is applied only to the
        # waveform returned to the model, not to the cached transcription.
        clean_waveform = self.load_audio_segment(audio_path, start_sec)
        transcription = self.get_transcription(
            audio_path,
            start_sec,
            clean_waveform,
        )
        transcription_tokens = self.get_transcription_tokens(transcription)

        waveform = self.process_waveform(clean_waveform)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return waveform, label_tensor, transcription_tokens

    def __len__(self):
        return self.num_files

    # -----------------------------------------------------------------
    # Optional DataLoader helper
    # -----------------------------------------------------------------

    def collate_fn(self, batch):
        """Stack fixed waveforms and pad variable-length token sequences.

        Use this method as ``collate_fn=dataset.collate_fn`` when constructing
        a DataLoader. It returns the same three logical objects as __getitem__:
        batched waveforms, labels, and padded transcription tokens.
        """
        waveforms, labels, transcription_tokens = zip(*batch)

        waveforms = torch.stack(waveforms)
        labels = torch.stack(labels)

        padding_value = self.tokenizer.pad_token_id
        if padding_value is None:
            padding_value = 0

        transcription_tokens = pad_sequence(
            transcription_tokens,
            batch_first=True,
            padding_value=padding_value,
        )

        return waveforms, labels, transcription_tokens