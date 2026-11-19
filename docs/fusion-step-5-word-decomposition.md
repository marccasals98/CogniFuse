# Step 5: regularized word decomposition and cross-modal interaction

**Implemented as an optional training path.** `--fusion_method decomposed_fixed`
constructs audio-private, text-private, shared, and cross-modal vectors for each
original word. Their fixed equal-weight average feeds recording pooling and
classification. The default `legacy` path retains the previous architecture.
The [Step 6 reliability inputs](fusion-step-6-reliability-features.md) and
[Step 7 learned gates](fusion-step-7-learned-gates.md) extend these same branches.

## Representation construction

[`scripts/word_fusion.py`](../scripts/word_fusion.py) receives aligned audio/text
adapter outputs `[B, N, input_dim]` and the paired word mask. Existing modality
adapters must have matching output widths; decomposition maps that common width
to `D` (default 256). Raw encoder widths can differ if their adapters map them
to the same width.

| Representation | Construction |
|---|---|
| $p_i^a$ | Audio-private projection of the acoustic word vector |
| $p_i^t$ | Independent text-private projection of the text word vector |
| $s_i^a, s_i^t$ | One shared projection, with the same parameters applied to audio and text |
| $s_i$ | Layer-normalized average $(s_i^a+s_i^t)/2$ |
| $C_i$ | Projected, layer-normalized concatenation of bidirectional cross-attention outputs |

Each private/shared projection is `Linear → Tanh → LayerNorm`. Merged shared
and cross-modal branches also use LayerNorm to control scale. `D` must be at
least 3 and divisible by the positive head count. The minimum avoids centered
two-dimensional nonzero vectors, which cannot satisfy the intended private/shared
orthogonality penalty.

The temporary fixed fusion is:

$$
z_i=\tfrac14(p_i^a+p_i^t+s_i+C_i).
$$

All branches and the fused tensor are `[B, N, D]`. Each valid word has four
fixed weights of `0.25`; invalid words have zero weights and vectors.

### Cross-attention direction and scope

Normalize $p_i^a+s_i^a$ and $p_i^t+s_i^t$ to form acoustic/text attention inputs.
Two independent multihead modules compute audio queries against text keys/values
and text queries against audio keys/values. Each query preserves its original
word position. Concatenating the directional results and projecting to `D`
defines $C_i$, without a direct query residual.

The default `--decomposition_cross_radius 1` permits keys at original positions
`i-1`, `i`, and `i+1` when valid. Unsupported/padded rows retain their positions
and cannot serve as keys. `-1` permits all valid recording words; zero and values
below `-1` are rejected. A window with only one valid key has no attention-ranking
choice. This mask defines scope, not sparse computation: the implementation
still uses dense attention masks. Encoder vectors retain their cached context.

## Learning objectives

Training uses the existing classification criterion plus four regularizers:

$$
\mathcal L=\mathcal L_{\mathrm{cls}}
+\lambda_a\mathcal L_{\mathrm{agreement}}
+\lambda_s\mathcal L_{\mathrm{separation}}
+\lambda_r\mathcal L_{\mathrm{reconstruction}}
+\lambda_v\mathcal L_{\mathrm{variance}}.
$$

| Term | Definition | CLI coefficient | Default |
|---|---|---|---|
| Shared agreement | Mean squared difference between $s_i^a$ and $s_i^t$ | `--shared_agreement_weight` | 0.1 |
| Private/shared separation | Average squared cosine similarities for $(p_i^a,s_i^a)$ and $(p_i^t,s_i^t)$ | `--private_shared_separation_weight` | 0.01 |
| Reconstruction | Separate linear decoders reconstruct each modality's adapter output from its private/shared concatenation | `--decomposition_reconstruction_weight` | 0.1 |
| Shared variance | Penalize shared coordinates whose across-word standard deviation is below 1 | `--shared_variance_weight` | 0.1 |

Squared errors average feature coordinates and, where applicable, modalities.
Agreement, separation, and reconstruction then average valid words within each
recording before averaging recordings, giving recordings equal weight.

Reconstruction targets are detached, clean adapter outputs before classification
dropout. Stop-gradient avoids a direct gradient through the reconstruction
target; adapters remain trainable through the encoding path. Decoders use their
own modality-specific shared vectors, rather than the merged shared vector.

Variance uses population variance over valid words independently per recording
and shared modality, then `relu(1 - sqrt(variance + 1e-4))`, averaged over
coordinates/modalities and recordings. Recordings with fewer than two valid
pairs contribute zero to this term because their variation cannot be estimated.

Coefficients must be finite and nonnegative; zero supports ablations. The stock
trainer requests all terms and keeps their paths connected even at zero weight.
It logs raw component values plus classification and weighted total loss. `loss`
is the total; validation predictions and checkpoint selection still use class
logits and the existing evaluation protocol.

These objectives **encourage** the intended split. Orthogonality is not
statistical independence, reconstruction does not identify a unique split, and
variance regularization is a soft penalty. Names and gradients alone do not
prove exclusively private/shared information or prevent collapse. These defaults
are initial engineering settings, not validated clinical hyperparameters;
evaluate objective ablations and representation behavior on development data.

## Enable the new path

After packaging every selected development recording:

```bash
.venv/bin/python scripts/train.py \
  --dataset adresso --dataset_root /path/to/ADReSSo21 \
  --number_classes 2 --num_folds 5 --fold 0 \
  --precomputed_features_dir /path/to/frame_packages/train \
  --precomputed_input_type frames \
  --acoustic_word_pooling attention \
  --fusion_method decomposed_fixed \
  --decomposition_dim 256 --decomposition_heads 4 \
  --decomposition_cross_radius 1 \
  --training_batch_size 1 --evaluation_batch_size 1
```

Mean acoustic pooling also works. The first version requires frame packages
with an explicit word axis; raw-audio and older tensor modes retain legacy
fusion. Existing packages need no regeneration for this architectural change.

The new module replaces legacy `seq_to_seq_method`; its width, heads, and scope
use decomposition options. `seq_to_seq_input_dropout` applies after fixed fusion
to keep auxiliary targets clean. Existing recording pooling, pooling dropout,
classifier, classification criterion, and optimizer remain active. Pooling
receives `N` fused positions and the paired mask, rather than concatenated
modality positions.

Only the real pilot has been packaged so far. Full-fold training requires the
remaining cohort packages. This step exercised the pilot and a small synthetic
cohort, rather than a full-cohort performance experiment.

## Inspect branches and use checkpoints

Default forward calls return logits for the existing evaluator. Inspection is
explicit:

```python
output = model(
    frame_batch=inputs,
    return_auxiliary=True,
    return_representations=True,
)
logits = output["logits"]
loss_terms = output["auxiliary_losses"]
branches = output["representations"]
```

Representations include the four branches, modality-specific shared vectors,
fused vectors, paired mask, fixed weights, and detached directional attention
maps `[B, heads, N, N]`. Auxiliary output also exposes decoder reconstructions.
Invalid queries have zero inspection weights. Internally, they use valid-key
fallback rows before being zeroed, avoiding all-masked softmax/NaN gradients.
Every recording must have at least one valid pair.

Requested branch tensors retain their graphs; the model caches no loss/branch
history. Custom training loops must request auxiliary losses and add their
coefficients. The stock trainer does this automatically and uses plain logits
for evaluation.

Checkpoint settings retain mode, dimension, heads, radius, and CLI loss weights.
New runs add a decomposition tag to their names. Older checkpoints default to
legacy fusion and retain their state-dict structure. New decomposition
checkpoints load strictly through the evaluator; switching architectures is
not an automatic old-checkpoint migration.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_word_fusion.py' -v
```

Ten focused tests cover shapes/fixed weights, local/global attention, invalid
queries, padding, per-recording averaging, agreement and variance behavior,
detached reconstruction targets, coefficient composition, zero-weight parameter
connectivity, classification gradients, checkpoint reload, and actual CLI
training/standalone evaluation.

The [pilot probe](audits/step5-adrso010-word-fusion-probe.json) produces four
`[1, 41, 256]` branches from 1,139 acoustic frames and 41 text words. All branch,
attention, and decoder groups, the acoustic scorer, and classifier receive
finite nonzero group gradients and update after one optimizer step. Attention
weights sum to one within approximately `1.2e-7` and are zero outside the local
neighborhood. Checkpoint tensors reload exactly; the pilot's reload and padding
logit errors are `0.0`. Tests allow normal small floating-point differences
between inference paths.

This fresh-model probe uses a synthetic target with frozen encoder tensors.
It establishes implementation behavior, not clinical performance or achieved
disentanglement. All 13 package-file hashes remain unchanged. CPU checks passed;
GPU performance and full-cohort representation behavior remain unmeasured.

The combined regression run passes 166 tests, excluding only the existing
ignored import-broken `test_batch_adresso_words.py` documented in Step 3.
[Step 6](fusion-step-6-reliability-features.md) now adds reliability extraction
and training-fold normalization. [Step 7](fusion-step-7-learned-gates.md) adds
conventional and reliability-aware learned four-way gates using these branches.
