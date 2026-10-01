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
New Whisper preprocessing runs additionally save `preprocessing/words/<uid>.json`
beside each word CSV. These sidecars contain the full Whisper output, the exact
cleaned words retained in the CSV with word IDs and probabilities, recording
timestamps, transcription configuration, source file information, and excluded
investigator intervals. Whole-recording timestamps are already absolute with
respect to the recording; there is no dataset window duration or stride.
Automatic language detection and all existing CSV outputs remain unchanged.

Older CSVs still work without sidecars. Full Whisper metadata cannot be recovered
from those CSVs alone. Sidecars are written when Whisper preprocessing is run;
the existing script still regenerates its CSV outputs on each run. The embedding
extractor continues reading CSVs and does not yet validate sidecar configuration.

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

## Export acoustic frames for alignment (Step 2)

The optional frame exporter processes one complete recording using the same
`facebook/wav2vec2-base-960h` acoustic encoder as the existing offline extractor.
It saves the full temporal sequence before word or token pooling. It runs
independently of Whisper and text embedding extraction; existing transcripts,
embeddings, masks, and training commands remain usable as before.

```bash
uv run python -m utils.prepro_audio_frames \
  --audio-path /path/to/WAB_samples/WAB_samples/recording.wav \
  --output-dir /path/to/WAB_samples/preprocessing/audio_frames/recording \
  --local-files-only
```

Choose a new output directory for each recording. Existing directories are
refused before loading models. Omit `--local-files-only` to allow a model
download; `--device cpu` or `--device cuda` overrides automatic device selection.
The command writes:

- `frames.pt`: CPU tensor `[T_frames, D_audio]` of unpooled hidden states.
- `valid_mask.pt`: boolean tensor `[T_frames]`. All entries are true because
  each recording is encoded individually without padding or augmentation.
- `metadata.json`: source identity, encoder and processor configuration,
  package versions, source/resampled sample counts, tensor shape, and timing.

Audio is mixed to mono and resampled to 16 kHz. Timing uses the encoder's
convolution kernels and strides and verifies the actual output length. For the
current encoder, consecutive frames are 20 ms apart and convolution intervals
are 25 ms wide. Frame 0 spans `[0, 0.025)` seconds and its center is 0.0125 s.
These intervals describe the convolution grid; transformer features also use
surrounding recording context. Timestamps use the original recording origin.

`utils.audio_frame_timing.frame_interval_seconds(index, metadata['timing'])`
returns a frame's interval with an exclusive end.
`time_to_frame_index(seconds, metadata['timing'])` returns the nearest frame
center, clipping recording edges to the first/last valid frame and rejecting
times outside the recording. A 40-second, 16-kHz recording has 1,999 frames;
using duration divided by frame count would only approximate the true stride.

This step supports the existing Wav2Vec2 encoder only. It does not perform CTC
alignment or change the classifier. Temporal adapters are explicitly rejected
because they need a different timing calculation.

Run the offline checks (synthetic audio and a small randomly initialized
Wav2Vec2 model, without downloads):

```bash
uv run python -m unittest discover -s tests -p 'test_audio_frames.py' -v
```

## Prepare Spanish CTC emissions (Step 3)

`scripts/ctc_encoder.py` loads the encoder and trained linear CTC head from
[`jonatasgrosman/wav2vec2-large-xlsr-53-spanish`](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-spanish).
The model card specifies Spanish speech recognition and 16-kHz audio. Its
character vocabulary includes Spanish accents and ñ. The CTC head is applied
only to its own encoder's hidden states. The Step 2 baseline features remain
available for downstream pooling; they are not passed to this different head.

Prepare the same test recording using its existing Whisper word CSV:

```bash
dataset_root=/home/usuaris/veussd/marc.casals/datasets/WAB_samples
recording=BISD008_6MoFU_PicnicDescription_Castellano
uv run python -m utils.prepro_ctc \
  --audio-path "$dataset_root/WAB_samples/$recording.mp3" \
  --words-csv "$dataset_root/preprocessing/words/$recording.csv" \
  --output-dir "$dataset_root/preprocessing/ctc/step3_first_recording"
```

The first run downloads the Spanish checkpoint unless it is already cached.
Use `--local-files-only` after that to require cached files. `--device cpu|cuda`
overrides automatic selection; `--revision` can pin a checkpoint commit.
Existing output directories are refused. Neither the CSV nor the Step 2 export
is modified. The JSON metadata from Step 1 is optional for this command.

The new directory contains `ctc_logits.pt` and `ctc_log_probs.pt`, each shaped
`[1, T_ctc, V_ctc]`, plus `valid_frame_mask.pt` shaped `[1, T_ctc]`,
`metadata.json`, and `transcript.json`. Metadata records the CTC vocabulary,
blank ID, model configuration/revision, source identities and frame timing.
The model stays frozen in evaluation mode. Loading an encoder-only checkpoint
that would randomly initialize a CTC head is rejected.

Transcript normalization preserves the exact original word list and indices.
It lowercases NFC text, removes punctuation, and preserves supported accents.
Every CTC character maps back to an original word; inter-word delimiter units
have word ID -1. Unsupported characters (including digits needing verbalization)
raise an explicit error with the word index. Punctuation-only entries remain
in the word mapping with `no_ctc_units` status instead of disappearing. Their
treatment in word pooling must be explicit in the later alignment step.

This prepares scores and transcript targets only. It does not compute word
boundaries, confidence or alignment entropy. Forced alignment is Step 4;
CTC frames and baseline acoustic frames must be matched through their timing
metadata rather than assuming frame indices are interchangeable.

Run the offline tests without downloading the Spanish checkpoint:

```bash
uv run python -m unittest discover -s tests -p 'test_ctc_encoder.py' -v
```

## Force-align cached CTC targets (Step 4)

Step 4 uses the installed TorchAudio 2.7
[`forced_align`](https://docs.pytorch.org/audio/2.7.0/generated/torchaudio.functional.forced_align.html)
implementation on CPU. It reads the Step 3 export without loading a speech
model, rerunning Whisper, or downloading anything:

```bash
dataset_root=/home/usuaris/veussd/marc.casals/datasets/WAB_samples
uv run python -m utils.ctc_forced_alignment \
  --ctc-dir "$dataset_root/preprocessing/ctc/step3_first_recording" \
  --output-dir "$dataset_root/preprocessing/ctc/step4_first_recording"
```

The new directory contains `path.pt` (the token ID at each valid frame),
`path_log_probs.pt` (selected per-frame log probabilities), and `alignment.json`.
The JSON records each target character/delimiter's frame span, time span,
original word ID, and emission score. It preserves the complete original word
mapping and hashes the input artifacts for provenance. Existing exports are
never overwritten or modified.

Frame spans use `[start_frame, end_frame_exclusive)`. Time spans cover the
associated convolution receptive fields, so adjacent character intervals can
overlap by the receptive-field/stride difference. The exporter validates the
timing grid, transcript mapping, normalized probabilities, exact CTC path
collapse, repeated-character constraints, and monotonic in-bounds token spans.
Only a contiguous valid prefix is aligned; trailing padded frames are excluded.

`emission_confidence` is `exp(mean_log_emission)` over the token's selected
nonblank frames. It is not a calibrated boundary posterior or alignment
entropy. Blank frames remain in the path and do not imply silence. Forced
alignment enforces transcript order but does not establish transcript or
boundary accuracy. Word interval merging and boundary inspection follow in
the next steps; Step 4 itself returns character/delimiter intervals only.

```bash
uv run python -m unittest discover -s tests -p 'test_ctc_forced_alignment.py' -v
```

## Merge CTC units into word windows (Step 5)

Step 5 reads the character alignment from Step 4 and creates one entry for each
original word, preserving its exact text, order and word ID. It requires no
models, GPU, downloads, or transcription reruns:

```bash
dataset_root=/home/usuaris/veussd/marc.casals/datasets/WAB_samples
uv run python -m utils.ctc_word_windows \
  --alignment-dir "$dataset_root/preprocessing/ctc/step4_first_recording" \
  --output-dir "$dataset_root/preprocessing/ctc/step5_first_recording" \
  --ctc-context-frames 0
```

The new directory contains `word_windows.json` and a readable
`word_windows.csv`. Existing output directories are refused, and the Step 4
input remains unchanged. The JSON includes its source hash, CTC timing/model
metadata, the original word list, one window per word, and a boolean
`word_alignment_mask` with the same length as the word list.

Each aligned word spans its first through last CTC character. Internal blank
frames are included in that interval; exterior blanks and word-delimiter
units do not extend the word's boundaries. `start_frame` and
`end_frame_exclusive` describe the original interval. These are **CTC encoder
frame indices**; later pooling into a different acoustic encoder must use the
timing metadata to map between grids.

`--ctc-context-frames` defaults to zero. For a margin of 2, 4, or 8 frames on
each side, choose a new output directory. The separate `expanded_start_frame`
and `expanded_end_frame_exclusive` fields describe the context window, clipped
to the valid frame sequence. Original boundaries remain available for comparing
with Whisper. Expanded windows may overlap, and their starts/ends remain
monotonic. Both original and expanded intervals include time fields using the
same convolution-receptive-field convention as Step 4.

Punctuation-only entries with no CTC characters retain their original position,
have `no_ctc_units` status and null intervals (empty cells in the CSV), and are
false in `word_alignment_mask`. No timestamps are fabricated. Corrupted word
ownership, missing characters, invalid timing, and out-of-bounds spans fail
explicitly. This step does not pool embeddings or assess boundary accuracy;
Step 6 will compare these word intervals with Whisper.

```bash
uv run python -m unittest discover -s tests -p 'test_ctc_word_windows.py' -v
```

## Patient-level cross-validation


The held-out fold is now selectable with `--fold` (zero-based, default `1`).
The default preserves the existing single-fold split. To run just another fold
with the CALCULA settings:

```bash
sbatch shs/calcula/train.sh --fold 0
```

Run all five folds sequentially on one GPU using the existing embeddings:

```bash
sbatch shs/calcula/train.sh --cross_validate
```

The runner starts a fresh model and optimizer in a separate process for each
fold, keeping the same split seed and training configuration. Every patient is
held out exactly once; all recordings from that patient stay together. The
current cohort has 202 recordings from 75 patients. There are 50, 42, 42, 43 and
25 validation recordings in folds 0 through 4, respectively. Patient grouping
means recording counts do not have to be equal.

CV evaluates the final model after the same fixed epoch budget in every fold
(`--max_epochs`, currently 10 in the launcher). It disables intermediate
validation-based checkpoint selection, early stopping and validation-driven
learning-rate changes. This keeps the held-out fold out of the training
decisions. Ordinary single-fold training retains its existing behavior. Compare
CV runs under the same protocol; these final-model scores differ from selecting
the best validation checkpoint. Existing checkpoints cannot initialize CV runs.

During CV training, logs report evaluation as pending and W&B omits unevaluated
metrics. After the final evaluation, W&B records `training_eval_metric` and
`validation_eval_metric`, also named `cv/final_training_macro_f1` and
`cv/final_validation_macro_f1`. CV runs omit `best_model_*` metrics because they
do not select a checkpoint by validation score. W&B's `loss` is the current
batch loss; `epoch_mean_batch_loss` is the arithmetic mean of batch losses over
the completed epoch (each batch has equal weight).

Results are written under `<log_file_folder>/cross_validation/<run_id>/`, or a
new directory specified by `--cross_validation_output_dir`. Files include:

- `splits.json`: patient and recording assignments for every fold.
- `config.json`: the CV configuration and fixed-epoch protocol.
- `fold_0.json` through `fold_4.json`: final validation predictions, macro-F1,
  per-class F1 and checkpoint paths.
- `fold_metrics.csv`: one row per fold.
- `summary.json`: mean and sample standard deviation of fold macro-F1, plus
  pooled out-of-fold macro/per-class F1 across all held-out recording predictions.

Scores are computed per recording; patients define the split boundary. Models
and training logs remain in the configured output directories, with fold indices
in their names. With W&B enabled, each fold creates its own run.

Check assignments without training or writing outputs:

```bash
uv run python scripts/train.py --cross_validate --cv_dry_run --max_epochs 10 \
  --train_labels_path /path/to/WAB_samples/labels.csv \
  --validation_labels_path /path/to/WAB_samples/labels.csv \
  --precomputed_features_dir /path/to/WAB_samples/preprocessing/embeddings_full
```

The sequential runner currently supports precomputed features and one process
(`--nproc_per_node=1`). It requires the same labels CSV for both splits and enough
patients of every selected class to populate all folds. Individual `--fold` runs
also support the raw-audio workflow.

## Repository organization
The main folders of the repo are the following:

* `/scripts`: The main scripts following classical PyTorch file structure.
* `/shs`: Scripts designed to launch experiments in HPC systems (using SLURM).
* `/utils`: Auxiliary stand-alone files that have no direct impact on the training.
