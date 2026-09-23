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
  --precomputed_features_dir /path/to/WAB_samples/preprocessing/embeddings_full
```

This selects `PrecomputedADDataset`: one paired audio/text tensor per recording,
with the same patient-level folds and class weights as the existing datasets.
Extraction processes the entire transcript in consecutive, non-overlapping
200-position encoder inputs. It then joins all content-token embeddings in
order, keeping only the first prefix and last suffix special tokens. No content
tokens are dropped or duplicated. Audio features use the original word
timestamps for every retained token, including words in later chunks.

The encoder's context is local to each chunk; downstream attention sees the
joined recording. Recordings remain single training examples regardless of
their number of chunks. Their sequence lengths may differ, including exceeding
the encoder's context limit, because the encoder is never run on the joined
sequence. Batches pad to the longest recording and use masks to ignore padding.
Longer sequences increase downstream attention memory and training time; start
with batch size 1 when comparing against the old 200-position baseline.

Generate the full-transcript variant from existing Whisper outputs:

```bash
sbatch shs/calcula/prepro_embeddings.sh --local-files-only
```

This writes to `preprocessing/embeddings_full`, preserving the original
`preprocessing/embeddings` baseline. Omit `--local-files-only` if the models are
not cached. The Python script also accepts `--dataset-path`, `--embeddings-dir`,
`--chunk-size`, and `--limit` (for smoke tests). Existing output files are refused
unless `--overwrite` is explicitly passed. A JSON sidecar per recording records
the encoder, chunk size, token count, and chunk count. Whisper does not need to
be rerun. Both text and aligned audio embeddings must be regenerated: mask
reconstruction cannot recover vectors for words discarded by the old extractor.

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
unused. Each tensor now requires a boolean mask beside it: `<uid>distil_mask.pt`
and `<uid>distil_audio_mask.pt` for the default filenames. Masks exclude padding
from attention and pooling. Text masks keep special tokens; audio masks keep
aligned tokens and the whole-recording summary at position zero, excluding
unused special-token positions such as SEP. Missing or invalid masks fail
explicitly. Both legacy fixed-length embeddings and new full-transcript
embeddings can be loaded, but legacy files still contain their original
truncation; point training at `embeddings_full` to use the complete sequences.

For legacy fixed-length embeddings extracted before masks were saved, reconstruct them using the
**original transcripts, tokenizer and extraction length**, without running
Whisper or either feature encoder:

```bash
uv run python -m utils.backfill_embedding_masks \
  --transcriptions /path/to/WAB_samples/preprocessing/transcriptions.csv \
  --embeddings-dir /path/to/WAB_samples/preprocessing/embeddings \
  --tokenizer distilbert-base-uncased --max-length 200 --dry-run
```

Remove `--dry-run` to write the masks after validation. Word timestamps are read
from `words/` beside the transcript CSV (override with `--words-dir`). The command
validates all pairs, checks alignment and audio row layout, preserves embedding
files, and refuses to overwrite conflicting masks. Already matching masks are
reused. Use `--local-files-only` to load a cached tokenizer without network access.
For other variants, set `--text-suffix`, `--audio-suffix`, and
`--transcript-column` (e.g. `transcription_pause`) to match the original extraction.
These checks cannot establish provenance if the original transcripts were replaced.

Omit `--precomputed_features_dir` to use the raw-audio workflow. Its existing text
mask is now also applied to downstream attention and pooling. Audio waveform
padding/crop behavior in that workflow is unchanged.

## Repository organization
The main folders of the repo are the following:

* `/scripts`: The main scripts following classical PyTorch file structure.
* `/shs`: Scripts designed to launch experiments in HPC systems (using SLURM).
* `/utils`: Auxiliary stand-alone files that have no direct impact on the training.
