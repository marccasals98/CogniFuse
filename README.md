# CogniFuse

Predict Cognitive decliness.

## How to use
CogniFuse uses uv to manage Python versions, dependencies, virtual environments, and reproducible installations.

Install uv and run the following command before executing:

```bash
uv sync --locked
```

The default `SimpleADDataset` uses a fixed pool of 8 evenly spaced crops per
training recording (`--crops_per_recording 8`). It still samples just one crop
per recording per epoch; validation uses a single center crop. Set the pool
size to 1 to use only the center crop for training too.

Before loading the classifier, training prepares missing crop transcripts with
Whisper and saves them in `--transcription_cache_dir`. This first pass can take
time, but subsequent epochs and runs reuse the files. Keep the cache directory
between runs. Whisper is released before classifier training begins. Changing
the pool size can introduce new crops that need transcription; changing the
window, sample rate, Whisper model, or language also changes the cache keys.
The pool limits crop diversity in exchange for bounded transcription work.
Trainable feature encoders still run during training.

To train from the outputs of `utils/prepro_whisper.py` followed by
`utils/prepro_embeddings.py`, pass the embeddings directory:

```bash
uv run python scripts/train.py \
  --train_labels_path /path/to/WAB_samples/labels.csv \
  --validation_labels_path /path/to/WAB_samples/labels.csv \
  --precomputed_features_dir /path/to/WAB_samples/preprocessing/embeddings
```

This selects `PrecomputedADDataset`: one paired audio/text tensor per recording,
with the same patient-level folds and class weights as the existing datasets.
The default filenames are `<uid>distil_audio.pt` and `<uid>distil.pt`, matching
the current preprocessing settings. For another preprocessing variant, set
`--precomputed_audio_suffix` and `--precomputed_text_suffix` explicitly (including
`.pt`). The UID is the recording filename without its directory or extension.
All selected recordings must have both files; missing pairs raise an error.

Feature dimensions are inferred from the tensors. Speech/text extractor options
are unused in this mode; the adapters, pooling and classifier remain trainable.
For different audio/text dimensions (e.g. mel features), configure adapters with
matching output dimensions. Raw audio, Whisper, tokenization and waveform
augmentation are skipped, and `--simple_dataset`, window and crop settings are
unused. The saved tensors already have a fixed sequence length; all positions
are retained because preprocessing does not save an attention mask. Omit
`--precomputed_features_dir` to use the existing raw-audio workflow.

## Repository organization
The main folders of the repo are the following:

* `/scripts`: The main scripts following classical PyTorch file structure.
* `/shs`: Scripts designed to launch experiments in HPC systems (using SLURM).
* `/utils`: Auxiliary stand-alone files that have no direct impact on the training.
