# Step 8: development cohort and matched fusion comparison

The complete 166-recording ADReSSo development cohort is packaged and
quality-ready under `outputs/reliability_fusion_cohort_v1/`. All five saved folds
pass package/schema and training-only normalization checks. No recording was
excluded. The 71 official test recordings are outside this experiment.

The comparison tooling prepares, executes, and audits three matched variants:
`decomposed_fixed`, `decomposed_gated`, and `reliability_gated`. Preparing the
experiment does not produce performance results. Final results require all
15 fold models to finish and pass their checkpoint audits.

The [cohort readiness receipt](audits/step8-development-cohort.json) records
652,603 acoustic frames, 17,758 original word rows, and 17,751 valid word pairs.
The seven unsupported acoustic words retain their original rows. It pins the
cohort and experiment manifest hashes and verifies the existing pilot packages
are unchanged.

## Prepare or verify the cohort

```bash
.venv/bin/python -m utils.prepare_fusion_cohort \
  --feature-root outputs/adresso_deberta_v3_large_wavlm_large_ctc_20261105 \
  --dataset-root /home/usuaris/veussd/marc.casals/datasets/ADReSSo21 \
  --output-root outputs/reliability_fusion_cohort_v1
```

Preparation audits the cached lineage, reconstructs frame-to-word means,
packages frozen frames/text without encoder inference, then extracts the
versioned Step 6 measurements. Independent pending and ready packages live in
`frames/train/<uid>` and `quality/train/<uid>` respectively. Original exports
and the earlier pilot packages are unchanged.

`--dry-run` checks development coverage and split-source identities without
writing packages. Normal execution holds a preparation lock, publishes packages
without overwriting them, journals completion receipts, and validates completed
packages on resume. Changed source configuration, saved splits, or completed
ready metadata fail explicitly. An interrupted partial package is never treated
as complete; inspect it before retrying or use a new destination.

After all records are ready, the utility tests normalization separately on each
training fold. It records observation counts and discards those fitted states.
Each actual training process fits and checkpoints its own state. The cohort
summary is a readiness receipt, not a global scaler.

| Fold | Train recordings | Validation recordings | Observed paired training words per word feature |
|---|---:|---:|---:|
| 0 | 132 | 34 | 14,375 |
| 1 | 133 | 33 | 14,014 |
| 2 | 133 | 33 | 14,090 |
| 3 | 133 | 33 | 14,282 |
| 4 | 133 | 33 | 14,243 |

All three recording measurements are observed for every training recording in
each fold. These counts refer to paired-valid words, not all stored word rows.

## Freeze the comparison

```bash
.venv/bin/python -m utils.compare_fusion_modes prepare \
  --cohort-root outputs/reliability_fusion_cohort_v1 \
  --experiment-root outputs/reliability_fusion_comparison_v1
```

The experiment manifest freezes package metadata receipts, saved splits,
configuration, source code, and the common initialization. Training runs from
the copied `source/` tree so subsequent repository edits cannot change queued
jobs. Preparation refuses an existing experiment root. Inputs are checked
again before a run, and result auditing verifies settings and split identities.

The first comparison uses a single Torch seed (`3407`) and the existing five
development folds. ADReSSo grouping uses recording UIDs; independent participant
identity is unverified in this distribution.

| Setting | Shared value |
|---|---|
| Encoders | Cached WavLM-large / DeBERTa-v3-large, pinned revisions |
| Acoustic word pooling | Learned attention, hidden width 64 |
| Adapters | Identity; 1024-wide cached inputs |
| Branches | Width 256, 4 cross-attention heads, radius 1 |
| Learned gate | Hidden width 64; conventional/quality input widths differ |
| Auxiliary coefficients | Agreement 0.1, separation 0.01, reconstruction 0.1, variance 0.1 |
| Recording pooling | Attention pooling |
| Dropout | 0.3 after fusion, before pooling, and in classifier |
| Classifier | Width 512, `classifier_hidden_layers=1` |
| Training | 60 epochs, batch size 1, AdamW, learning rate 0.0001, weight decay 0.01 |
| Loss | Cross-entropy with training-fold class weights |
| Selection | First eligible checkpoint, then strict validation macro-F1 improvements, every 100 updates and at the terminal model |
| Early stopping / LR changes / augmentation | Disabled |

The budget, optimizer, pooling, classifier, and dropout follow the earlier T24
development configuration. This is a new architecture comparison, not a replay
of T24. Branch and acoustic-pooling settings are the Step 4–7 engineering
defaults, shared across variants. Neither gate width nor regularization is
tuned separately for one mode.

The historical CLI extractor tags in run names are placeholders when cached
features are used. Actual encoder identities and revisions come from the pinned
frame-package schema; no encoder is instantiated during these training runs.

Gate construction preserves the common Torch RNG stream. For a given seed,
all common initial weights, including the classifier, match across modes; the
manifest records the identical common-state hash. Gate parameter counts are
0, 65,860, and 66,756; total model parameter counts for this configuration are
2,965,378, 3,031,238, and 3,032,134 respectively.

## Execute and inspect

For one variant in the current environment:

```bash
.venv/bin/python -m utils.compare_fusion_modes run \
  --experiment-root outputs/reliability_fusion_comparison_v1 \
  --variant reliability_gated
```

`--dry-run` verifies pinned inputs and exercises CV split planning without
training. Existing variant directories are refused to protect partial or
completed runs. Failures create `failed.json` and retain training logs.

On the configured Slurm cluster:

```bash
.venv/bin/python -m utils.compare_fusion_modes submit \
  --experiment-root outputs/reliability_fusion_comparison_v1
```

Submission creates three array tasks with at most two concurrent GPUs and a
dependent CPU summary job. Each array task trains its five folds sequentially.
The manifest, `submission.json`, generated scripts, per-variant logs, checkpoints,
and fold predictions remain local to the experiment. W&B is disabled. Repeated
submission or submission after local training has begun is rejected.

The initial full comparison was submitted on November 19, 2026 as array
`2721771`, with dependent summary job `2721772`. Results are pending; these job
IDs record submission, not successful completion. Inspect Slurm status and the
experiment's `comparison.json` for current progress.

```bash
.venv/bin/python -m utils.compare_fusion_modes summarize \
  --experiment-root outputs/reliability_fusion_comparison_v1 \
  --require-complete
```

The audit checks settings, exact splits and held-out UIDs, checkpoint decision
journals, strict checkpoint loading, and the normalization's training UIDs and
package receipts. It reconstructs CV summaries from fold results and checks
paired recording/label identities across variants. Complete results include
mean/sample-SD fold macro-F1, pooled out-of-fold macro-F1, per-class scores,
selected epochs, parameter counts, and paired fold-score differences in
`comparison.json` and `comparison.csv`. An incomplete run has explicit pending
variants and cannot claim a complete comparison.

These are development scores: the same held-out fold selects its checkpoint
and supplies its reported score. A single-seed difference does not establish
statistical significance, independent-test generalization, or degradation
robustness. Step 9 remains the controlled corruption evaluation.

## Verification

Eight focused tests cover exact source-fold matching, development/test
separation, resumable preparation and receipts, common initialization/RNG
parity, frozen-source/package changes, incomplete-run behavior, bounded and
duplicate-safe submission, and actual three-mode CV on a tiny two-fold fixture.
The integration test audits every checkpoint and rejects normalization whose
training UIDs have been changed.

The combined CPU regression run passes 194 tests, excluding only the existing
ignored import-broken `test_batch_adresso_words.py` documented in Step 3. The
real fixed and conventional-gate jobs have also initialized on CUDA and entered
training; completing those experiments is separate from the passing tests.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest \
  discover -s tests -p 'test_fusion_comparison.py' -v
```

The tiny fixture validates experiment execution and auditing; its scores are
not evidence about ADReSSo performance. The real cohort receipt validates input
readiness. Performance claims require the completed real comparison.
