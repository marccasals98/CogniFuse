# Step 6: reliability features and training-fold normalization

**Status: implemented.** Packages can now contain four word-level quality
measurements and three recording-level measurements, with explicit availability
masks. Training fits normalization on its selected training fold and saves the
state in the checkpoint. Validation and evaluation reuse that state.

The current fixed four-way fusion does not use these measurements to change
predictions. The learned fusion gates are the next step.

## Extracting a quality-ready package

Starting from a validated Step 2 package:

```bash
.venv/bin/python -m utils.extract_fusion_quality \
  --package-dir outputs/reliability_fusion_frames_v1/train/adrso010 \
  --output-dir outputs/reliability_fusion_quality_v1/train/adrso010
```

This command has already produced the pilot package above. Use a new destination
for another run: existing directories are never overwritten. The extractor
locates the original waveform through the package's SHA-256 provenance. If it
has moved, supply `--audio-path /path/to/the/same/audio.wav`; its content hash
must still match. No speech or text encoder runs during enrichment.

The extractor copies the eight existing feature, interval, ID, and validity-mask
files byte for byte, writes four quality tensors, and publishes metadata last.
It checks that source files have not changed during extraction. The resulting
package is independently readable without the original audio or source package.
Its reader checks tensor hashes, schemas, ranges, masks, and word-quality values
against the saved evidence. Waveform measurements are not recomputed on load.

The versioned ready schema is `fusion_quality_v1`, with recording estimator
`energy_quiet_spectrum_v1`. Feature order and estimator settings contribute to
the package schema fingerprint. Unknown versions, orderings, or settings fail
validation. Historical pending packages with `Q=0, R=0` remain supported;
ready packages use `Q=4, R=3`.

## Raw measurements

The extractor joins the existing Whisper/CTC evidence by original word ID and
checks the word strings. The ordered word features are:

| Position | Feature | Definition |
|---|---|---|
| 0 | `asr_confidence` | Saved Whisper word probability, in `[0, 1]` |
| 1 | `ctc_emission_confidence` | Exponentiated frame-weighted mean character log emission |
| 2 | `boundary_start_disagreement_seconds` | Absolute CTC/Whisper start-time difference before context expansion |
| 3 | `boundary_end_disagreement_seconds` | Absolute CTC/Whisper end-time difference before context expansion |

For CTC units with frame durations `n_j` and mean log emissions `l_j`, the word
measurement is `exp(sum(n_j * l_j) / sum(n_j))`. Delimiters and blanks do not
contribute. This measures support for the supplied transcript; it is not a
calibrated probability that the word or its boundaries are correct. Boundary
disagreement uses the saved absolute differences, not a boundary posterior.

The ordered recording features are:

| Position | Feature | Definition |
|---|---|---|
| 0 | `snr_proxy_db` | Energy contrast `10 * log10(p90 / p10)`, clipped to `[0, 60]` dB |
| 1 | `clipping_ratio` | Fraction of original channel samples with `abs(sample) >= 0.999` |
| 2 | `spectral_flatness_noise_proxy` | Spectral flatness of the mean quiet-frame power spectrum |

Clipping is measured on decoded samples before channel mixing, DC removal, or
gain changes. For the other two measurements, channels are averaged, the
recording mean is removed, and complete nonoverlapping 20 ms frames are used;
any trailing partial frame is discarded. `p10` and `p90` are the frame-power
percentiles, each floored at `1e-12` for the ratio.

Quiet frames are the lowest-power `max(2, ceil(0.1 * frame_count))` frames.
Apply a nonperiodic Hann window, average their squared FFT magnitudes, discard
the DC bin, and add `1e-12` per bin. Spectral flatness is the geometric mean
divided by the arithmetic mean of this spectrum, clipped to `[0, 1]`.

At least ten complete frames are required for energy and noise measurements.
Energy contrast is unavailable if `p90 <= 1e-12`; spectral flatness is
unavailable if mean quiet-frame power is at most `1e-12`. A silent recording
therefore has missing energy/noise measurements and an observed clipping ratio
of zero. A short recording still has an observed clipping ratio.

These are energy and noise proxies, not calibrated SNR or reverberation
estimates. Speech dynamics and content can affect them. Boundary entropy and
reverberation estimation remain deferred. Pause duration, speech rate, jitter,
and shimmer are absent from this vector.

## Missingness and normalization

Raw tensors contain finite values and Boolean observation masks. An unavailable
measurement uses a stored zero with `observed=False`; a measured zero keeps
`observed=True`. Quality availability does not change word validity or filter
words using quality thresholds.

Use the existing frame input options with a directory of ready packages:

```text
--precomputed_input_type frames
--precomputed_features_dir /path/to/reliability_fusion_quality_v1/train
```

Training automatically fits the following transformation after selecting the
training-fold dataset:

1. Compute each word feature's population mean and standard deviation over
   observed, paired-valid training words only.
2. Compute each recording feature's statistics once per training recording,
   considering observed values only. Recordings with more words do not receive
   additional weight for these statistics.
3. Replace a standard deviation below `1e-6` with `1`. If a feature has no
   observed training values, fail explicitly and name it.
4. Standardize observed values and impute missing transformed values with zero,
   equivalent to imputing the training mean before standardization.
5. Broadcast recording features to words and append availability indicators.

The normalized per-word vector has 14 columns, in this order:

```text
[word z-scores (4), recording z-scores (3),
 word observation flags (4), recording observation flags (3)]
```

Invalid paired-word rows and batch padding are entirely zero. Raw values and
masks remain available alongside `reliability_features` in dictionary batches.

The checkpoint stores feature order, means, scales, observation counts, selected
training UIDs, and training metadata hashes. Validation and test datasets apply
that saved state without fitting. Resuming training requires the same selected
training records and metadata receipts; changed folds or packages fail instead
of silently refitting. Model inputs must carry the matching normalization-state
fingerprint. W&B configuration summaries omit the per-record receipts; the full
state remains in checkpoint settings.

A ready-package checkpoint needs its saved normalization state for evaluation.
Historical pending-package checkpoints retain their original behavior and schema;
they cannot silently switch to the ready schema. Only the real pilot package
has been enriched so far: a cohort run requires ready packages for all selected
recordings, with statistics fitted separately inside each training fold.

## Verification

The [pilot report](audits/step6-adrso010-quality-probe.json) records extraction on
`train/adrso010`: `[41, 4]` word measurements, all observed, and three observed
recording measurements. Its energy contrast is about `46.680` dB, clipping
ratio `0.0181613`, and quiet-spectrum flatness `0.00147096`.

A one-record training-only smoke fixture produces `[1, 41, 14]` normalized
features and reproduces them exactly after checkpoint-state reload. This is a
normalization mechanics check, not cohort cross-validation statistics. The
original pending package remains unchanged, and copied feature files match
byte for byte.

Ten focused tests cover extraction, missingness, identity checks, controlled
noise perturbations, fold isolation, recording weighting, checkpoint replay,
resume receipts, and trainer/evaluation integration. The combined CPU regression
run passes 176 tests, excluding only the existing ignored import-broken
`test_batch_adresso_words.py` documented in Step 3. GPU behavior and clinical
performance have not been measured by this step. Controlled degradation and
subgroup analyses remain part of the later gate evaluation.
