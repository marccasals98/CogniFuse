# Frame-based fusion input contract

**Status: Step 1 design, ready for the package exporter in Step 2.** These files
are the planned training interface, not an export format already consumed by
the classifier. The [pilot audit](fusion-step-1-audit.md) establishes which
existing artifacts can supply them.

## Preprocessing and training boundary

Preprocessing runs the frozen speech/text encoders, transcription, and forced
alignment. It retains unpooled acoustic hidden states, text word vectors, word
intervals, and measurement-quality sources. Training learns acoustic word
pooling, representation branches, cross-attention, fusion, recording pooling,
and classification.

Speech features are whole-recording contextual encoder outputs, not independent
raw-frame measurements. Attention restricted to a word window weights those
contextual vectors; it does not remove context already introduced by the encoder.
Text subword mean pooling remains in preprocessing for this first version.

Start with ADReSSo development recordings from the WavLM-large/DeBERTa-large
export. The contract permits other encoder dimensions and cohorts. Labels,
participant identities, partitions, and development folds remain in their
existing dataset index, separate from predictive features. The pilot choice is
not evidence that the full cohort has passed this audit.

## Per-recording package

Use a new format identifier, `fusion_frame_inputs_v1`, and one directory per
recording under a separate export root. Store each recording's acoustic frames
once, rather than copying frame windows for every word. Use CPU tensors loaded
with `weights_only=True`.

Let `T` be the saved acoustic frame count, `N` the original word count, `A` and
`E` the acoustic/text encoder widths, `Q` the word-quality feature count, and
`R` the recording-quality feature count. These sizes are inferred and validated,
not hard-coded from the pilot.

| File | Type and shape | Meaning |
|---|---|---|
| `frames.pt` | float32 `[T, A]` | Frozen speech encoder `last_hidden_state`, before word pooling |
| `frame_mask.pt` | bool `[T]` | Valid saved frames; no batch padding in the export |
| `word_ids.pt` | int64 `[N]` | Exact original word IDs `0 .. N-1` |
| `word_frame_intervals.pt` | int64 `[N, 2]` | Start-inclusive, end-exclusive indices into `frames.pt` |
| `audio_word_mask.pt` | bool `[N]` | Whether the word has a usable acoustic interval |
| `text_words.pt` | float32 `[N, E]` | Frozen text word vectors, mean-pooled over their subwords |
| `text_word_mask.pt` | bool `[N]` | Whether the word has a usable text vector |
| `paired_word_mask.pt` | bool `[N]` | Acoustic AND text validity; initial fusion/pooling policy |
| `word_quality.pt` | float32 `[N, Q]` | Raw word-level measurement-quality values |
| `word_quality_observed.pt` | bool `[N, Q]` | Which quality values were actually available |
| `recording_quality.pt` | float32 `[R]` | Raw recording-quality values, broadcast during training |
| `recording_quality_observed.pt` | bool `[R]` | Availability of each recording-level measurement |
| `metadata.json` | JSON | Format, identities, word order, timing, feature definitions, and file hashes |

### Word identity and validity

- Preserve the exact original words and their order in metadata. Repeated words
  are distinct rows; text alone is never a join key. This interface has no
  added CLS/SEP rows.
- Retain unsupported, punctuation-only, and otherwise invalid words. An invalid
  acoustic interval is `[-1, -1]` with `audio_word_mask=False`. Never index frames
  using that sentinel. Invalid text vectors are zero with `text_word_mask=False`.
- Every valid interval satisfies `0 <= start < end <= T` and covers valid frames.
  Empty intervals are invalid. Preserve the existing explicit nearest-center
  fallback when mapping a short CTC interval to another acoustic frame grid.
- Require `paired_word_mask == audio_word_mask & text_word_mask` and at least
  one valid pair per recording. The initial implementation pools and classifies
  valid pairs, matching the current cohort policy.
- Missing confidence does not invalidate an otherwise usable embedding.
  Conversely, a quality value cannot make an invalid interval valid.

### Timing and word windows

Use the mapping in [`utils/word_audio_pool.py`](../utils/word_audio_pool.py).
CTC and acoustic encoders are independent: map through their recorded timing
when their grids differ. Use timestamps relative to the original recording;
never estimate the grid from duration divided by frame count.

Store original CTC intervals, mapped acoustic intervals, mapping method, source
timing, and context configuration in metadata. The first package uses the
existing zero-context windows. Any later expansion is an explicit export or
model configuration change, with clipping at recording edges. Reusing the same
stored full frame sequence permits new windows without rerunning the encoder.

The first attention module will learn weights inside a fixed word window.
Learning word-boundary positions is a separate extension.

### Reliability schema

Define feature names and units in an ordered schema; the model checks that
schema rather than assuming a fixed width. The first available word-quality
schema is:

| Position | Name | Definition |
|---|---|---|
| 0 | `asr_confidence` | Whisper CSV `probability`, in `[0, 1]` |
| 1 | `ctc_emission_confidence` | Exponentiated frame-weighted mean character log emission, excluding delimiters and blanks |
| 2 | `boundary_start_disagreement_seconds` | Absolute CTC minus Whisper start-time difference, before context expansion |
| 3 | `boundary_end_disagreement_seconds` | Absolute CTC minus Whisper end-time difference, before context expansion |

The CTC definition is given in the [gate design](reliability-aware-fusion.md).
It measures support for the supplied transcript, not a probability that its
boundaries are correct. The pilot audit derives it as a feasibility check;
the later extractor must implement and version this definition.

An early package used only to verify mean/attention pooling may use `Q=0` and
`R=0`, with empty named schemas and quality status `pending`. Once quality
extraction is enabled, emit the four named word features above (`Q=4`). Keep
`R=0` until recording-quality estimators have been defined and implemented.
A model configured for reliability-aware fusion must reject a pending or
incompatible schema. Changing the feature list changes its schema fingerprint
and requires matching model configuration.

Later recording features can include a specified SNR proxy, clipping ratio, and
noise/reverberation proxy. Their estimators, thresholds, units, and waveform
preprocessing must be recorded before adding them to the schema. Boundary
entropy is absent until a boundary posterior and entropy definition exist.
Pause duration, speech rate, jitter, and shimmer are excluded from this vector.

Store all tensor values as finite numbers. Use zero as the storage placeholder
for unobserved quality and set its observation mask false; observed zero remains
distinguishable. During training, fit scaling/imputation on observed values in
valid training words/recordings only. Apply the saved transformations to held-out
data, impute missing transformed values, and append availability indicators to
the gate input. If a requested feature has no observations in the training fold,
fail explicitly or remove it through a recorded schema change.

Fit recording-level transformations once per training recording rather than
once per repeated word. Save their state and feature order with the checkpoint.

### Provenance

Metadata must contain:

- Format/schema identifiers, recording UID, partition, ordered words and IDs,
  tensor shapes, dtypes, counts, and the mask policy.
- Actual acoustic, text, and CTC encoder IDs and pinned revisions, acoustic
  layer/context/precision, text chunking, and subword pooling configuration.
- Acoustic and CTC convolution timing, interval conventions, context settings,
  and per-word mapping methods.
- Named quality schemas, estimator definitions/versions, and extraction status.
- Source audio, word CSV, and intermediate file identities with SHA-256 hashes;
  relative package paths and hashes for each tensor in the new package.

The Step 2 exporter will publish independent packages without modifying existing
feature exports. It must validate inputs before publishing and reject conflicting
outputs. Existing intermediate paths can be retained as provenance; training
must not require those paths to read a completed package.

## Batching and training invariants

The loader pads frame and word axes independently. Frames/text/quality values
pad with zero, interval and word-ID tensors with `-1`, and all masks with false.
Stack recording features to `[B, R]`; broadcast only when assembling word-level
gate inputs. Validate that every recording in a batch has the same encoder and
quality schema, while allowing different frame/word counts.

For the first integration, a dictionary batch carries features, intervals,
modality masks, quality values/availability, recording UIDs, and labels. UIDs
and labels are metadata/supervision and never enter the gate. The existing
precomputed-word dataset path keeps its current interface.

Acoustic pooling returns `[B, N, A]`. Apply the existing normalization and
adapters after raw-frame mean pooling in the compatibility baseline. Normalizing
frames before averaging would change the baseline. Pass only `N` word positions
and their paired mask to recording pooling, not the old concatenated `2N` mask.

The encoder tensors remain frozen; gradients must reach learned pooling and
fusion parameters. Training must not precompute and reuse outputs of those
trainable modules between optimizer steps.

## Acceptance checks for the next steps

1. New-package mean pooling reproduces the old valid word vectors exactly, or
   within documented floating-point tolerance after a deliberate implementation
   change. Compare original masks and IDs as well as numerical values.
2. The existing classifier with identical weights reproduces baseline logits
   through the new mean-pooling path in evaluation mode.
3. Different-length batches, repeated words, invalid intervals, and missing
   quality are handled explicitly; padded frames/words cannot affect predictions.
4. Learned attention normalizes over valid frames and receives classification
   gradients. Gate weights normalize across the four available branches.
5. Checkpoint reload restores architecture, feature schemas, preprocessing
   transformations, and predictions. Evaluation uses the same package interface.

These are checks for the exporter/model work ahead. Step 1 has verified only
the existing pilot artifacts and reconstruction described in the audit report.
