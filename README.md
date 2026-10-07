# CogniFuse

CogniFuse classifies recordings into cognitive-language diagnostic groups using
speech and text features. The current WAB experiment uses `lvPPA`, `nfPPA`, and
`svPPA` labels and keeps recordings from the same patient in the same fold.

## Choose a data workflow

There are two main ways to supply data: extract features during training, or
save features beforehand and train from those files. The CTC + MrBERT experiment
uses the second approach and reuses the existing classifier.

| Workflow | What one training example contains | How audio and text are matched | What runs during training |
|---|---|---|---|
| Raw audio: recording crops (`SimpleADDataset`) | One sampled crop per recording per epoch | Whisper transcribes the crop | Speech/text encoders, adapters, fusion, pooling, classifier |
| Raw audio: overlapping windows (`ADDataset`) | One fixed window; a recording can produce several examples | Whisper transcribes each window | Speech/text encoders, adapters, fusion, pooling, classifier |
| Precomputed Whisper-aligned features | One recording's saved token-level audio/text tensors | Whisper word times associate audio with text tokens | Adapters, fusion, pooling, classifier; encoders are skipped |
| Precomputed CTC-aligned word features | One recording's saved audio/text word tensors | Spanish CTC refines word times; acoustic frames and text subwords are averaged per word | Same existing adapters, fusion, pooling, classifier; encoders are skipped |

For precomputed features, changing the classifier does not require extracting
embeddings again. Changing the encoder, transcript, alignment, or extraction
settings can require new exports. Keep different variants in separate directories.
Both precomputed variants use `PrecomputedADDataset`; the directory and filename
suffixes tell it which features to read. A mask marks valid rows so padding and
unusable word pairs do not contribute to attention or pooling.

- [Setup](#setup)
- [Raw-audio training](#raw-audio-training)
- [Precomputed Whisper-aligned features](#precomputed-whisper-aligned-features)
- [CTC + MrBERT: single-recording steps 1–9](#ctc-word-alignment-step-by-step)
- [CTC + MrBERT: whole-dataset preprocessing](#ctc-whole-dataset-preprocessing)
- [Train and evaluate](#train-and-evaluate)
- [Outputs and training summaries](#outputs-and-training-summaries)

## Setup

CogniFuse uses `uv` to manage Python and dependencies. From the repository root:

```bash
uv sync --locked
```

The CALCULA launchers contain paths for the current WAB installation. Examples
using `/path/to/...` are placeholders that must be replaced with real paths.
For the CTC examples below, set these variables in your shell:

```bash
dataset_root=/home/usuaris/veussd/marc.casals/datasets/WAB_samples
recording=BISD008_6MoFU_PicnicDescription_Castellano
```

`--local-files-only` requires models already cached on your machine. Omit it
for an initial download. The single-recording CTC exporters require a new
output directory; the batch runner supports resuming completed work.

## Raw-audio training

Do not pass `--precomputed_features_dir` for this workflow. Use `--simple_dataset`
for recording crops or `--no-simple_dataset` for the older overlapping-window
workflow (`--window_secs` and `--stride_secs`). The CALCULA training launcher
already supplies a precomputed directory, so use the Python entry point when
selecting raw audio, for example:

```bash
uv run python scripts/train.py \
  --train_data_dir /path/to/audio \
  --validation_data_dir /path/to/audio \
  --train_labels_path /path/to/labels.csv \
  --validation_labels_path /path/to/labels.csv \
  --simple_dataset
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

## Precomputed Whisper-aligned features

If word transcripts are missing, first run
[Step 1](#prepare-whisper-transcripts-and-word-ids-step-1). This workflow uses
Whisper timestamps directly; it does not require CTC Steps 2–9.

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

## CTC word alignment step by step

This sequence explains and checks the pipeline on **one recording**. It prepares
features; none of Steps 1–9 trains the classifier. To prepare the full cohort,
use the batch workflow after checking a representative recording.

| Step | Input → output | Purpose |
|---|---|---|
| 1 | Audio → Whisper word CSV | Establish the words and their original order |
| 2 | Audio → acoustic frames | Save unpooled speech features |
| 3 | Audio + words → CTC scores and targets | Prepare Spanish speech alignment |
| 4 | CTC scores + targets → character intervals | Force-align the supplied transcript |
| 5 | Character intervals → word windows | Define one interval per alignable word |
| 6 | CTC windows + Whisper times → comparison reports | Inspect boundary disagreements |
| 7 | Acoustic frames + word windows → audio word vectors | Average frames within each word |
| 8 | Original words → text word vectors | Encode with MrBERT and average subwords |
| 9 | Audio/text word vectors → loader files | Package matching tensors and masks |

Steps 2 and 3 both read the audio, but use different encoders for different
purposes. Step 7 uses the acoustic features from Step 2 and the boundaries from
Step 5. Step 6 is a review checkpoint. Packaging one recording in Step 9 only
checks compatibility; training needs every selected recording's package.

### Prepare Whisper transcripts and word IDs (Step 1)

If `preprocessing/words/<uid>.csv` already exists for your recordings, reuse it.
The current batch runner starts at Step 2 and requires these word CSVs.
For a dataset without transcripts, the existing WAB launcher runs Whisper:

```bash
sbatch shs/calcula/prepro_whisper.sh
```

This script processes the dataset, not just the example recording. Its dataset
path and Whisper `turbo` model are configured in `utils/prepro_whisper.py`.
It writes `preprocessing/transcriptions.csv`, per-recording word CSVs, and JSON
metadata. Word CSVs contain `word`, `start`, `end`, and `probability`; their row
order defines the original word IDs preserved by the CTC pipeline. Existing
investigator segmentation is used when available.

**Do not rerun this just to continue a later step:** this older script regenerates
its transcript/word outputs. Preserve existing transcripts when reproducing an
export. Missing JSON sidecars alone do not prevent CTC preprocessing.

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

### Export acoustic frames for alignment (Step 2)

The optional frame exporter processes one complete recording using the same
`facebook/wav2vec2-base-960h` acoustic encoder as the existing offline extractor.
It saves the full temporal sequence before word or token pooling. It runs
independently of Whisper and text embedding extraction; existing transcripts,
embeddings, masks, and training commands remain usable as before.

```bash
uv run python -m utils.prepro_audio_frames \
  --audio-path "$dataset_root/WAB_samples/$recording.mp3" \
  --output-dir "$dataset_root/preprocessing/audio_frames/step2_first_recording" \
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

### Prepare Spanish CTC emissions (Step 3)

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

### Force-align cached CTC targets (Step 4)

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

### Merge CTC units into word windows (Step 5)

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

### Compare CTC word boundaries with Whisper (Step 6)

Step 6 prints an original-word-order table of Whisper intervals, unexpanded CTC
intervals, CTC frame ranges, and absolute start/end differences. It reads the
original Whisper CSV path from the Step 5 metadata. No models or GPU are needed:

```bash
dataset_root=/home/usuaris/veussd/marc.casals/datasets/WAB_samples
uv run python -m utils.compare_word_alignment \
  --word-windows-dir "$dataset_root/preprocessing/ctc/step5_first_recording" \
  --output-dir "$dataset_root/preprocessing/ctc/step6_first_recording" \
  --disagreement-threshold 0.2
```

The new directory contains `comparison.txt`, `comparison.csv`, and
`comparison.json`. Omit `--output-dir` to print without saving. Existing output
directories are refused. Inputs remain unchanged. The reader verifies the
Step 4 source hash, recomputes the Step 5 windows to check their consistency,
and checks the original Whisper CSV's saved size/modification time and exact
word sequence. Changed inputs fail explicitly rather than silently comparing
unrelated transcripts. Report metadata records input hashes.

`delta_start` and `delta_end` are absolute differences in seconds; signed
differences are also saved as CTC minus Whisper. Added context is excluded from
all comparisons. Repeated words remain separate rows by original word ID.
Words without CTC intervals remain in the table with null differences and are
excluded from aggregate statistics. Whisper probabilities are retained when
available, separately from boundary differences.

An asterisk flags words whose start or end difference exceeds the configurable
threshold (default 0.2 seconds). This is a review aid, not a validated pass/fail
criterion or evidence that either method is correct. Summaries report mean,
median and maximum differences and the original IDs of flagged words. CTC time
spans still describe convolution support, which also affects small boundary
differences. Inspect the largest discrepancies before proceeding to pooling.

```bash
uv run python -m unittest discover -s tests -p 'test_compare_word_alignment.py' -v
```

### Mean-pool acoustic word embeddings (Step 7)

After reviewing the Step 6 boundary comparison, pool the saved Step 2 acoustic
frames using the Step 5 CTC word windows. This command runs on CPU and requires
no models, downloads, or raw-audio decoding:

```bash
dataset_root=/home/usuaris/veussd/marc.casals/datasets/WAB_samples
uv run python -m utils.word_audio_pool \
  --audio-frames-dir "$dataset_root/preprocessing/audio_frames/step2_first_recording" \
  --word-windows-dir "$dataset_root/preprocessing/ctc/step5_first_recording" \
  --output-dir "$dataset_root/preprocessing/word_embeddings/step7_first_recording"
```

It saves three files in a new directory, refusing to overwrite prior exports:

- `audio_word_embeddings.pt`: float32 tensor `[N_words, D_audio]`, one row per
  original word. Each valid row is the mean of the selected acoustic frames.
- `word_mask.pt`: boolean tensor `[N_words]`, marking valid pooled words.
- `metadata.json`: original words/IDs, CTC and acoustic frame intervals, timing,
  context margin, mapping method, source identities, and input hashes.

The selected intervals are the Step 5 **expanded** windows, so zero context
uses exactly the original boundaries and nonzero context uses the previously
configured margin. All frame ranges have exclusive ends. Matching timing grids
preserve the exact CTC frame slices. For different grids, pooling selects
acoustic frames whose centers lie inside the CTC convolution-support interval.
If no center lies inside a short interval, it selects the valid frame nearest
the interval midpoint and records `nearest_center_fallback` explicitly.

Source audio identities and recording durations must agree. Valid masks and
timing grids are checked; trailing padded acoustic frames are excluded. An entry
without CTC units keeps its original position and receives a zero vector with
a false word mask, never a fabricated acoustic window. Consumers must apply
the mask. Repeated words retain separate IDs and vectors.

The test recording produces `[83, 768]`, with all 83 words valid. These new
artifacts are separate from existing training inputs; the current classifier
is unchanged. Step 8 produces matching word-level text embeddings.

```bash
uv run python -m unittest discover -s tests -p 'test_word_audio_pool.py' -v
```

### Encode matching text word embeddings (Step 8)

Use [`BSC-LT/MrBERT`](https://huggingface.co/BSC-LT/MrBERT) explicitly for
the selected text-embedding experiment. MrBERT is a multilingual ModernBERT
encoder with 768-dimensional hidden states and an 8192-token context limit.
Its fast tokenizer maps subwords back to the same original word IDs used by
the acoustic export. This uses the existing Step 8 exporter and keeps BETO
and other compatible models available through `--text-model`.

```bash
dataset_root=/home/usuaris/veussd/marc.casals/datasets/WAB_samples
uv run python -m utils.prepro_word_text \
  --audio-words-dir "$dataset_root/preprocessing/word_embeddings/step7_first_recording" \
  --output-dir "$dataset_root/preprocessing/word_embeddings/step8_mrbert_first_recording" \
  --text-model BSC-LT/MrBERT \
  --local-files-only
```

The checkpoint is cached in the development environment. On another machine,
omit `--local-files-only` for the first download. The installed Transformers
version must support ModernBERT. Keep the current 512-token chunk setting for
this first comparison; MrBERT's larger context can be selected explicitly with
`--chunk-size 8192` in a separate export. Changing the chunk size can change
word vectors because it changes the available context.

The 83-word test recording tokenizes to 109 content subwords in one chunk.
Its expected text shape is `[83, 768]`, matching the acoustic word count.
The saved model revision and tokenizer hash distinguish this export from
other text encoders. All existing acoustic and text exports are preserved;
use a fresh output directory for every new export.

#### Text export details and the optional BETO variant

The following command selects the generic BETO default. It is an alternative
to the MrBERT export above, not an extra required step. Step 9 below uses the
MrBERT directory; change that input path if you choose BETO.

Step 8 reads the exact original word list from the Step 7 acoustic export and
encodes it with a Hugging Face fast tokenizer using `is_split_into_words=True`.
Subwords are grouped with `word_ids()` and mean-pooled into one text vector per
original word. The generic exporter default is
[`dccuchile/bert-base-spanish-wwm-uncased` (BETO)](https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased),
a Spanish BERT model. Existing text preprocessing and its model defaults are
unchanged. Select another compatible encoder with `--text-model` if needed.

```bash
dataset_root=/home/usuaris/veussd/marc.casals/datasets/WAB_samples
uv run python -m utils.prepro_word_text \
  --audio-words-dir "$dataset_root/preprocessing/word_embeddings/step7_first_recording" \
  --output-dir "$dataset_root/preprocessing/word_embeddings/step8_first_recording" \
  --local-files-only
```

BETO is cached in the current environment. Omit `--local-files-only` on a machine
where it needs downloading. `--device cpu|cuda` overrides automatic device
selection, and `--revision` can pin a checkpoint commit. Inference uses evaluation
mode without gradients. Existing output directories are refused before loading
the model. Acoustic exports remain untouched.

The new export contains:

- `text_word_embeddings.pt`: float32 `[N_words, D_text]` tensor in the exact
  same word order as `audio_word_embeddings.pt` from Step 7.
- `text_word_mask.pt`: boolean validity for text rows.
- `paired_word_mask.pt`: intersection of the original audio mask and text mask.
- `metadata.json`: original words and IDs, per-word subword IDs/strings/counts,
  chunk ownership, model configuration/revision, tokenizer hash, input hashes,
  and both modality shapes.

Added special tokens and padding never enter word means. Punctuation belonging
to an original word stays attached to that word through `word_ids()`. An entry
with no text tokens keeps a zero row and false text mask. A punctuation-only
entry may have valid text but no acoustic interval; its paired mask is false.
Unknown subword counts are recorded rather than silently replacing word IDs.

`--chunk-size` defaults to 512 tokens including special tokens. Longer
transcripts are split at whole-word boundaries, with no overlapping chunks or
truncated tokens. Each word's complete subword sequence is encoded in one
chunk; context is local to that chunk. A single word exceeding the available
token budget fails explicitly. The joined token IDs and original word IDs are
checked against tokenizing the entire word list, ensuring full ordered coverage.

The exporter verifies `audio_rows == text_rows == len(words)` and preserves
the original word ID at each row. These artifacts prepare aligned inputs;
Step 9 packages them for the existing classifier; it does not create a new classifier.

```bash
uv run python -m unittest discover -s tests -p 'test_word_text_encoder.py' -v
```

### Package word embeddings for training (Step 9)

The word-level experiment reuses `PrecomputedADDataset` and `Classifier`,
including the existing adapters, attention, pooling, classification head, and
patient-based folds. Package a Step 7 acoustic export and its matching Step 8
text export into the loader's recording-based file layout:

```bash
dataset_root=/home/usuaris/veussd/marc.casals/datasets/WAB_samples
uv run python -m utils.package_word_embeddings \
  --audio-words-dir "$dataset_root/preprocessing/word_embeddings/step7_first_recording" \
  --text-words-dir "$dataset_root/preprocessing/word_embeddings/step8_mrbert_first_recording" \
  --output-dir "$dataset_root/preprocessing/word_embeddings/step9_first_recording"
```

The new directory contains `<uid>_word_audio.pt`, `<uid>_word_text.pt`, their
`<feature_stem>_mask.pt` files, and `<uid>_word_metadata.json`. The UID comes
from the original audio filename, matching the label CSV naming convention.
Packaging verifies original words and IDs, Step 8's acoustic input hashes,
source identities, finite tensors, shapes, and mask consistency before writing.
An existing destination is refused. No model weights are loaded by packaging.

Both loader masks use `audio_mask & text_mask`: this experiment includes only
valid audio/text pairs. Every original word row and vector remains present;
the original modality masks, paired mask, source hashes, and output hashes are
recorded in the sidecar metadata. The existing loader reads the tensors and
masks; it does not consume the sidecar. Keep it with the files for auditing.
No special-token rows or recording-summary vectors are added.

When the complete training cohort has been exported into a shared directory,
the existing trainer can select it with `--precomputed_features_dir` plus
`--precomputed_audio_suffix _word_audio.pt` and
`--precomputed_text_suffix _word_text.pt`. The trainer infers each modality's
feature dimension from the saved tensors. Keep the original full labels CSV
and patient folds for training; this one-recording directory is only for
checking compatibility and is not a complete training dataset.

This bridge preserves the current fusion behavior. In the CALCULA launcher,
`MultiHeadAttention` attends over separate audio/text positions and
`AttentionPooling` summarizes them before `ClassifierLayer`. Packaging does
not introduce concatenation within each word or restrict attention to matched
word pairs. Such a fusion comparison would be a separate, optional addition.

```bash
uv run python -m unittest discover -s tests -p 'test_package_word_embeddings.py' -v
```

## CTC whole-dataset preprocessing

Use this once the single-recording pipeline is understood. Both launchers run
Steps 2–9 from existing word CSVs, collect training files, and validate the
patient folds. They do not train the classifier.

Choose **strict** processing to reject words outside the Spanish CTC vocabulary,
or explicitly choose **mask unsupported words** to keep recordings while
excluding those word pairs from classifier input. These are two policies for
the same pipeline, not two stages to run consecutively.

### Strict batch preprocessing

Submit the remaining recordings on a CALCULA GPU:

```bash
sbatch shs/calcula/prepro_ctc_words.sh
```

This runs the existing Steps 2–9 for the target `lvPPA`, `nfPPA`, and `svPPA`
recordings in the full labels CSV. It reuses the validated packaged recording
from `step9_first_recording` and does not load an encoder for that recording.
Existing preprocessing and training launchers are unchanged. No training job
is launched by this command.

To inspect inputs without inference or writing exports:

```bash
bash shs/calcula/prepro_ctc_words.sh --dry-run
```

The batch root is `preprocessing/ctc_mrbert_batch` under the WAB dataset.
Use `--output-root /path/to/a/new/batch` to select another experiment directory.
Inside it, `recordings/` holds stage attempts, `embeddings/` holds the flat files
for the existing training loader, and `reports/` holds input errors, per-stage
failures, and a summary for each invocation. Slurm output is written to
`ctc_mrbert_words_<jobid>.out` in the repository.

Models are loaded once per stage, with only one encoder on the GPU at a time.
The runner pins checkpoint revisions and retains the existing 512-token text
chunks, zero-frame CTC context, and whole-recording acoustic inference. Long
recordings may exceed GPU memory; such errors are reported rather than changing
the signal or silently truncating it. Resubmit the same command to resume.
For failed inference on a GPU, `--device cpu` can resume the unfinished stages,
at a higher runtime cost.

Successful stages have checksummed completion records. Resume verifies those
records and their dependencies before reusing them. Incomplete attempts remain
intact; retries use a new attempt directory. A lock prevents concurrent writers
to the same batch root. Collection creates independent copies and refuses
conflicting existing files. Changed model/settings or label cohorts require a
new output root. Source hashes invalidate affected stages when source files
change, and collection still refuses to replace previously published exports.

The audit on 2026-10-04 found all 202 audio files and word CSVs, but **30
recordings contain unsupported CTC characters** (77 word rows, including digits,
Catalan accents, and other scripts). The Spanish CTC normalizer intentionally
rejects these. The batch processes eligible recordings and reports the remaining
ones in `reports/*_input_issues.csv`; it never rewrites transcripts or silently
reduces the training cohort. To include these recordings, review the transcripts or explicitly choose the
unsupported-word masking policy below. A nonzero exit status is expected while any recording
remains unresolved; completed work remains available for resumption.

The final report includes per-recording CTC/Whisper boundary disagreement
summaries for review. Disagreement is not an automatic exclusion criterion.
`training_ready` becomes true only when every target recording is packaged and
the unchanged loader successfully reads the complete cohort, with disjoint
patients in every training/validation fold and each recording held out once.
This is a completeness check, not a claim of alignment accuracy. For a ready
batch, use its `embeddings/` directory with the `_word_audio.pt` and
`_word_text.pt` suffixes documented above.

```bash
uv run python -m unittest discover -s tests -p 'test_batch_word_embeddings.py' -v
```

### Opt-in: mask unsupported CTC words and retain all recordings

The strict character checks remain the default. To keep all original word IDs
while omitting unsupported words from the CTC targets, submit:

```bash
sbatch shs/calcula/prepro_ctc_words_skip.sh
```

This launcher enables `--skip-unsupported-words`, uses the new batch root
`preprocessing/ctc_mrbert_skip_words`, and validates/reuses the 172 completed
packages from `preprocessing/ctc_mrbert_batch/embeddings`. It processes the 30
previously blocked recordings and collects the complete cohort under the new
root's `embeddings/` directory. Existing exports, transcripts, and the strict
batch plan remain intact. Resubmit the same launcher to resume this experiment.

An input-only check is also available:

```bash
bash shs/calcula/prepro_ctc_words_skip.sh --dry-run
```

For the current cohort, the opt-in audit accepts 202 recordings and identifies
77 unsupported word rows in 30 recordings. It skips the **whole word** when
any character is unsupported; it does not guess a replacement or retain only
the supported letters of a corrupted word. Punctuation-only entries continue
to follow the existing no-CTC-units policy and are not counted in these 77.
Malformed timestamps, internal whitespace, missing inputs, and recordings with
no remaining alignable words still fail explicitly.

Skipped words retain their original text and index throughout the exports.
They have no CTC targets or acoustic interval, a zero acoustic vector, and a
false acoustic mask. Text encoding still uses the original transcript and can
produce a text vector for the row; the paired mask is false. Packaging applies
that paired mask to **both** modalities, so the existing classifier ignores
those rows. No row is deleted and the audio/text arrays keep matching lengths.

Metadata records `unsupported_word_policy`, `skipped_word_ids`, and
`skipped_words` with the original strings and reasons. This audit is propagated
through emissions, alignment, word windows, comparison, acoustic/text pooling,
and packaging. The batch summary also records the total skipped-word count.
Alignment validation recomputes the permitted omissions from the full CTC
vocabulary instead of accepting an arbitrary list of ignored word IDs.

This policy masks word pairs rather than removing portions of the waveform or
text context. Omitted speech can affect neighboring CTC boundaries, and skipped
text remains part of the encoder context. The existing boundary-comparison
reports remain available. Training readiness still requires all 202 packaged
recordings and successful validation by the existing loader and patient folds;
this launcher does not start training.

For single-recording CTC preparation, `utils.prepro_ctc` also accepts
`--skip-unsupported-words`. The later commands automatically honor and validate
the recorded omission policy. Strict batches reject reused exports with
skipped words; skip-enabled batches can reuse unchanged strict exports.

```bash
uv run python -m unittest discover -s tests -p 'test_ctc_skipped_words.py' -v
```

## Train and evaluate

### Train using CTC + MrBERT word embeddings

After the batch report says `training_ready: true`, train one fold with:

```bash
sbatch shs/calcula/train.sh \
  --precomputed_features_dir /home/usuaris/veussd/marc.casals/datasets/WAB_samples/preprocessing/ctc_mrbert_skip_words/embeddings \
  --precomputed_audio_suffix _word_audio.pt \
  --precomputed_text_suffix _word_text.pt \
  --fold 1
```

To evaluate all five patient folds, use the same features and suffixes:

```bash
sbatch shs/calcula/train.sh \
  --precomputed_features_dir /home/usuaris/veussd/marc.casals/datasets/WAB_samples/preprocessing/ctc_mrbert_skip_words/embeddings \
  --precomputed_audio_suffix _word_audio.pt \
  --precomputed_text_suffix _word_text.pt \
  --cross_validate
```

The launcher otherwise defaults to `preprocessing/embeddings_full`, the
Whisper-aligned variant. Running it without the CTC directory and suffix
arguments does **not** select the CTC + MrBERT experiment. For a strict CTC batch,
substitute `ctc_mrbert_batch/embeddings` after checking its readiness report.

### Patient-level folds and cross-validation protocol

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
(`--max_epochs`, currently 60 in the launcher; override it explicitly for a different budget). It disables intermediate
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

## Outputs and training summaries

Yes: training records metrics per run, and the full cross-validation runner
also writes structured summaries. Preprocessing reports describe feature
completeness and alignment; they contain no classifier performance scores.

| Output | Location with the current CALCULA launchers | What to read |
|---|---|---|
| CTC training inputs | `$dataset_root/preprocessing/ctc_mrbert_skip_words/embeddings/` | Audio/text `.pt` tensors, masks, and per-recording metadata |
| Intermediate preprocessing | `<batch_root>/recordings/<uid>/<stage>/<attempt>/` | Saved stage outputs; reused packages can refer to the earlier batch root |
| Preprocessing reports | `<batch_root>/reports/` | `*_summary.json` for readiness, failures, skipped words, and alignment comparisons |
| Preprocessing Slurm output | Repository root | `ctc_mrbert_words_<jobid>.out` or `ctc_mrbert_skip_<jobid>.out` |
| Training logs | `/home/usuaris/veussd/marc.casals/logs/cognifuse/train/` | Per-run `.log` files with progress and measured metrics |
| Training Slurm output | `/home/usuaris/veussd/marc.casals/logs/sbatch/ser2025/` | `train_<jobid>.txt` with job output and errors |
| Model checkpoints | `/home/usuaris/veussd/marc.casals/models/<model_name>/` | `<model_name>.chkpt` |
| Cross-validation summaries | `/home/usuaris/veussd/marc.casals/logs/cognifuse/train/cross_validation/<run_id>/` | `summary.json`, `fold_metrics.csv`, and individual `fold_*.json` |

For an ordinary single-fold run, use its training log and, when enabled, its
Weights & Biases run for loss and training/validation macro-F1. The CALCULA
launcher enables W&B. A normal single-fold run does not automatically create
the cross-validation `summary.json` or a combined table of all experiments.

For a completed `--cross_validate` run, start with `fold_metrics.csv` to compare
folds and `summary.json` for mean ± sample standard deviation and pooled
out-of-fold scores. Individual fold JSON files also include predictions and
checkpoint paths. The combined summary is written only after all folds finish;
if a run stops early, inspect the completed fold files and logs.

## Verification

The per-step test commands above run focused checks. Run the complete suite with:

```bash
uv run python -m unittest discover -s tests -v
```

## Repository organization

The main folders of the repo are the following:

* `/scripts`: The main scripts following classical PyTorch file structure.
* `/shs`: Scripts designed to launch experiments in HPC systems (using SLURM).
* `/utils`: Preprocessing, alignment, packaging, and other utilities. Their outputs can be used as training inputs.
