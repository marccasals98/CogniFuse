# Step 3: frame packages in the training framework

**Implemented.** The optional `--precomputed_input_type frames` path loads
Step 2 packages, batches their frame and word axes independently, and performs
acoustic word mean pooling inside the classifier. The default remains `words`,
including for older checkpoints without this setting.

This step establishes a compatible trainable-model boundary.
[Step 4](fusion-step-4-acoustic-attention.md) adds learned frame attention;
the four-branch gate and quality extraction follow later.

## Loading and batching

[`scripts/data.py`](../scripts/data.py) provides frame-package variants of the
existing PPA and ADReSSo datasets. They reuse the current label maps, patient
folds for PPA, and saved development folds for ADReSSo. Every selected UID must
have a package at `<precomputed_features_dir>/<uid>/metadata.json`. Missing
packages, incorrect UIDs/partitions, and incompatible schemas fail explicitly.
Official ADReSSo test recordings stay unlabeled and require an explicit test
package directory.

Package headers establish the common schema; full tensor hashes and structural
checks run when an item is loaded. The loader retains quality tensors and their
availability masks, which remain empty/pending in the current package format.

[`scripts/frame_inputs.py`](../scripts/frame_inputs.py) assembles dictionary
batches containing:

- Frames `[B, T, A]` and frame validity `[B, T]`.
- Word intervals `[B, N, 2]`, original word IDs `[B, N]`, text `[B, N, E]`,
  and acoustic/text/paired word masks `[B, N]`.
- Quality values/availability, a common schema fingerprint, UIDs, and labels.

Frame and word padding are independent. Values pad with zero, intervals/IDs
with `-1`, and masks with false. UIDs and labels stay outside model inputs.
The device-transfer helper preserves floating, integer, and boolean dtypes.

## Model and trainer

The frame-mode classifier's mean baseline computes raw acoustic means within
valid word intervals before applying the existing normalization, adapters, attention,
recording pooling, and classifier. Invalid intervals are skipped; valid
intervals cannot include invalid or out-of-range frames.

Both modality masks supplied to the existing attention path use paired validity,
matching the previous packaged-word experiment. Its existing attention output
layout and pooling masks are preserved for this compatibility stage. A future
four-branch fusion output will use its own single word-axis mask.

The mean-pooling module has no parameters or state-dict entries, so the same
classifier weights can be compared directly. Classification still trains the
existing adapters, attention, and classifier; saved encoder tensors stay frozen.
The initial implementation reduces each word window separately to preserve the
old floating-point reduction order. GPU performance has not been benchmarked.

Optimization, training-set evaluation, and validation now share one batch-forward
helper. Frame schema/configuration are pinned into checkpoint settings and
checked against validation and inference packages. Older tuple batches and
saved word-feature checkpoints retain their existing input interface.

## Training commands

After exporting packages for **every selected development recording**, use the
existing training entry point, for example:

```bash
.venv/bin/python scripts/train.py \
  --dataset adresso \
  --dataset_root /path/to/ADReSSo21 \
  --number_classes 2 --num_folds 5 --fold 0 \
  --precomputed_features_dir /path/to/frame_packages/train \
  --precomputed_input_type frames \
  --training_batch_size 1 --evaluation_batch_size 1
```

The dimensions are inferred from loaded features. Other architecture, optimizer,
and checkpoint-selection options remain those of the existing trainer. For
PPA, supply the existing label CSV arguments and `--dataset ppa`. The sequential
CV runner also routes through the frame dataset factory and checks fold schemas.

Only `adrso010` has been exported into the real Step 2 package directory so far.
That single package cannot populate a complete ADReSSo training/validation fold;
the loader will reject missing cohort packages. No full-cohort training was
launched in this step.

## Standalone evaluation

The evaluator selects its reader from checkpoint settings. For a checkpoint
trained in frame mode:

```bash
.venv/bin/python scripts/evaluate.py \
  --checkpoint /path/to/frame_checkpoint.chkpt \
  --dataset_to_evaluate /path/to/frame_packages/test \
  --output_dir /path/to/new_evaluation_output \
  --batch_size 1 --device cpu
```

A single recording-package directory is also supported. Directory evaluation
requires complete packages from a single partition and a common checkpoint
schema. Labels remain optional and separate, as in the existing evaluator.
Recorded inference inputs are rechecked for changes before results are saved.
Inference requires the package files, not their old preprocessing directories.

## Verification

The [real-pilot parity report](audits/step3-adrso010-model-parity.json) compares
the existing trained T24 fold-0 checkpoint through both input paths on
development recording `adrso010`. It uses the same learned weights and evaluation
mode. Acoustic means, classifier logits, and logits after a frame-configured
checkpoint reload all match **exactly**, with maximum absolute error `0.0`.
No encoders were rerun or clinical performance estimates produced.

Focused checks:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_frame_inputs.py' -v
```

The tests cover saved development folds and unlabeled test data, PPA patient
separation, missing packages, different-length padded batches, invalid frames,
schema/partition mismatches, old/new logit parity, padding invariance, gradients,
checkpoint reload, standalone evaluation, and the actual training CLI on a small
synthetic cohort for one epoch with checkpoint and fold-result output.

Full `tests/` discovery also encounters a pre-existing import error in the
ignored, untracked `tests/test_batch_adresso_words.py`: it imports `PROFILES`
and other functions that the committed `utils.batch_adresso_words` does not
define. That file was left untouched. The other 135 tests in that discovery
passed, and the subsequently added PPA integration case passed in the focused
six-test run.

[Step 4](fusion-step-4-acoustic-attention.md) now provides optional learned
attention over each word's saved acoustic window and retains this mean path as
the comparison baseline.
