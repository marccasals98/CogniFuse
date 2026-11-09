# Step 2: frame-based package exporter

**Implemented and verified on the Step 1 pilot.**
[`utils/package_fusion_frames.py`](../utils/package_fusion_frames.py) exports
the `fusion_frame_inputs_v1` interface described in the
[input contract](fusion-input-contract.md). It runs the strict source audit,
reuses frozen encoder outputs, and packages one recording at a time.

[Step 3](fusion-step-3-training-inputs.md) now connects these packages to the
trainer and evaluator through an optional frame input mode with in-model mean
pooling.

## Create a package

From the repository root:

```bash
.venv/bin/python -m utils.package_fusion_frames \
  --feature-root outputs/adresso_deberta_v3_large_wavlm_large_ctc_20261105 \
  --uid adrso010 \
  --output-dir outputs/reliability_fusion_frames_v1/train/adrso010
```

This exact pilot directory has already been created. Use a new output directory
to repeat the export; existing destinations are always refused. `--partition`
defaults to `train` and also accepts `test`. The exporter uses the existing
ADReSSo batch summary and its complete stage lineage; it requires those sources
during export, including explicit acoustic checkpoint revision metadata.

The package contains 12 tensors and a completion manifest, `metadata.json`:

- One full acoustic frame sequence and its validity mask.
- Original word IDs and timing-aware acoustic frame intervals.
- Text word vectors, separate modality masks, and their intersection.
- Empty word/recording quality tensors and matching observation masks.
- A schema fingerprint, encoder identities, timing, original/mapped windows,
  text mappings, source hashes, and the raw ASR/CTC/boundary evidence.

Original feature files are read only. The exporter serializes independent copies
in a temporary sibling directory and validates them before publication. New
files are linked from that temporary directory, with `metadata.json` published
last. These links never point to original feature files. If publication is
interrupted, a directory without metadata is incomplete and must not be loaded;
choose a new destination for a retry. Temporary staging files are cleaned up.

## Read and validate a completed package

```python
from utils.package_fusion_frames import read_fusion_package, mean_word_vectors

metadata, tensors = read_fusion_package(
    "outputs/reliability_fusion_frames_v1/train/adrso010"
)
frames = tensors["frames.pt"]
intervals = tensors["word_frame_intervals.pt"]
audio_means = mean_word_vectors(tensors)  # Verification helper, not a training layer.
```

The reader checks tensor hashes, shapes/dtypes, finite values, the schema
fingerprint, original IDs, validity masks, invalid-word sentinels, timing-based
interval mapping, and validity counts. It uses only package files. Absolute
source paths in provenance do not need to exist when reading a completed package.
Relative tensor filenames remain fixed even if the package directory moves.

Quality extraction is deliberately pending: `word_quality.pt` is `[N, 0]`, and
`recording_quality.pt` is `[0]`. No missing quality value is being presented as
observed zero. The raw evidence is preserved in `metadata["quality_sources"]` for
the later quality extractor. Recording-quality estimators and boundary entropy
have not been added, and the reader currently accepts only this pending schema.

## Pilot verification

The new pilot package is under
`outputs/reliability_fusion_frames_v1/train/adrso010/` and contains approximately
5.02 MB across 13 files. The
[verification report](audits/step2-adrso010-frame-package.json) records its metadata
hash, schema fingerprint, counts, and checks.

| Item | Result |
|---|---|
| Acoustic frames | `[1139, 1024]`, exactly equal to original frames |
| Text word vectors | `[41, 1024]`, exactly equal to original vectors |
| Word intervals | `[41, 2]`, in acoustic frame indices with exclusive ends |
| Valid words | 41 audio, 41 text, 41 paired |
| Reconstructed acoustic means | Exactly equal to existing word vectors; maximum absolute error `0.0` |
| Masks | Exactly equal to the original frame/modality/paired masks |
| Portable read | A copied package loads with identical metadata and tensors |
| Original sources | All 45 audited source-file hashes remain unchanged |
| Encoder inference | None |

This packaging check verifies one recording, not full-cohort readiness.
Classifier parity is recorded separately in the Step 3 guide.

## Focused checks

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_package_fusion_frames.py' -v
```

Tests cover different CTC/acoustic grids, repeated words, invalid acoustic/text
rows, pending quality with missing raw confidence, exact mean reconstruction,
source independence, relocation after source removal, source/tensor tampering,
schema/interval validation, overwrite refusal, and interrupted publication.
