# Step 4: learned acoustic word attention

**Implemented.** Frame-input models can learn how to pool acoustic frames
inside each aligned word window. Mean pooling remains the default baseline.
Both modes use the same frozen encoder outputs and Step 2 package format.

## Computation

For acoustic hidden state $H_j$ and fixed word window $W_i$:

$$
e_j = v^\top\tanh(W H_j+b),
\qquad
\alpha_{ij}=\frac{\exp(e_j)}{\sum_{k\in W_i}\exp(e_k)},
\qquad
a_i=\sum_{j\in W_i}\alpha_{ij}H_j.
$$

The scoring MLP is `Linear(A, H) → Tanh → Linear(H, 1, bias=False)`.
The last bias is omitted because a common score offset cancels in softmax.
The default hidden width is 64: with 1024-wide acoustic features, this adds
65,664 trainable parameters. Standard linear-layer initialization is used;
attention is not initialized to reproduce the mean exactly.

Softmax runs over frames within each word, independently for every window,
including overlapping windows. Values are the original encoder vectors. The
resulting `[B, N, A]` word tensor feeds the existing normalization, adapters,
cross-modal sequence layer, recording pooling, and classifier.

The first scorer is acoustic-only. Text conditioning, reliability inputs,
positional features, and learned boundaries are separate extensions. The saved
vectors already contain encoder context; windowed attention limits which
contextual vectors are weighted, not their original receptive context.

## Enable attention

For a complete development package directory:

```bash
.venv/bin/python scripts/train.py \
  --dataset adresso --dataset_root /path/to/ADReSSo21 \
  --number_classes 2 --num_folds 5 --fold 0 \
  --precomputed_features_dir /path/to/frame_packages/train \
  --precomputed_input_type frames \
  --acoustic_word_pooling attention \
  --acoustic_word_attention_hidden_dim 64 \
  --training_batch_size 1 --evaluation_batch_size 1
```

Use `--acoustic_word_pooling mean` for the comparison run, keeping other settings
matched. Existing packages need no regeneration. Only one real recording has
been packaged so far; complete cohort packages are required before launching
development-fold training.

Attention is rejected outside frame-input mode, and its dimensions must be
positive integers. Pooling mode and hidden width are saved with checkpoint
settings and reported in CV configuration. The existing optimizer includes the
registered scorer parameters. Classification loss trains the scorer and
downstream trainable layers while encoder tensors remain frozen.

Older checkpoints without the new settings keep mean pooling and their original
state-dict structure. Attention checkpoints add `acoustic_word_pooling.scorer.*`
parameters and load strictly through the same standalone evaluator. Switching
from a saved mean model to attention is an explicit architecture change; the
evaluator does not silently initialize missing attention parameters.

## Masks and inspectable weights

Invalid/padded word rows are skipped and return zero vectors. Valid windows
must contain at least one acoustic frame and cannot include invalid/padded
frames. Out-of-window frames never reach the scorer. This avoids a dense
`[B, N, T]` attention tensor and all-masked softmax rows. A single-frame window
has weight one and offers no frame-weighting choice.

For an attention-mode model, request detached weight vectors explicitly:

```python
with torch.inference_mode():
    audio_words, rows = model.acoustic_word_pooling(
        batch["frames"],
        batch["word_frame_intervals"],
        batch["audio_word_mask"],
        batch["frame_mask"],
        return_attention=True,
    )
```

Each row contains `batch_index`, `word_index`, `start_frame`,
`end_frame_exclusive`, and `weights`. Weights correspond to consecutive acoustic
frame indices in that half-open interval. Diagnostics cover audio-valid words;
apply paired validity when summarizing words used by the classifier. The module
retains no weight history or computation graphs between batches.

These weights allocate contribution within an acoustic window. They differ
from the later four-way modality gate and are not calibrated quality
probabilities or a complete explanation of predictions.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_acoustic_word_attention.py' -v
```

Six focused tests check locality, normalization, overlapping windows, padding
invariance (including NaNs in unused padding), uniform-score mean equivalence,
single-frame identity, empty rows, and invalid configuration/ranges. They also
check finite nonzero scorer gradients, optimizer updates, zero gradients on
unused frames, classification gradients with frozen inputs, and detached
diagnostics. An actual CLI training run on a tiny synthetic cohort verifies
attention checkpoint settings/state and reproducible standalone evaluation.

The combined regression run passed 156 tests across `tests/` and
`evaluation_tests/`. It excluded only the existing ignored
`test_batch_adresso_words.py` module with the import failure documented in
the [Step 3 guide](fusion-step-3-training-inputs.md).

The [pilot probe](audits/step4-adrso010-attention-probe.json) uses the real
`adrso010` package and existing T24 classifier weights. Only the new scorer is
trainable. One synthetic least-likely-class target checks gradient flow and an
optimizer update; this is not a clinical training experiment or performance
estimate. Missing keys when copying baseline weights are checked to be exactly
the new scorer parameters.

The pilot has 1,139 frames and 41 audio-valid windows, 40 containing multiple
frames. Maximum weight-sum error is approximately `1.2e-7`; gradients reach
every scorer parameter. Strict reload reproduces logits exactly, the mean
baseline still reconstructs old word vectors exactly, and source features and
the original checkpoint remain unchanged.

Fixed windows cannot select evidence entirely outside their boundaries. GPU
speed/memory and cohort-level benefits have not been measured. The per-window
implementation prioritizes correct masking and a clear baseline; scalability
can be evaluated in the representative-recording pilot stage.

The next architectural step is private/shared representation decomposition and
a defined word-indexed cross-modal interaction, followed by reliability
extraction and the four-way gate.
