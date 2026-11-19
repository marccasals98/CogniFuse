# Step 7: learned four-way fusion gates

**Status: implemented.** The existing word branches now support conventional
and reliability-aware learned gates before masked recording pooling. Training
learns their weights through the classification objective and retains the
Step 5 auxiliary objectives. Default `legacy` fusion and optional fixed fusion
remain available.

## Modes and computation

| `--fusion_method` | Weights | Gate inputs |
|---|---|---|
| `legacy` | Existing architecture | Existing audio/text path |
| `decomposed_fixed` | Four equal weights | No gate |
| `decomposed_gated` | Learned per word | Four representation branches |
| `reliability_gated` | Learned per word | Four branches plus normalized quality features and availability flags |

Branch order is always `audio_private`, `text_private`, `shared`, `cross_modal`.
The learned modes concatenate the four `[B, N, D]` branches, appending the
Step 6 vector `[B, N, 14]` only for `reliability_gated`:

```text
h = [audio_private; text_private; shared; cross_modal; optional_quality]
weights = softmax(Linear_4(tanh(Linear_hidden(h))), dim=-1)
fused = sum(weights[..., branch] * branch_representation)
```

`--fusion_gate_hidden_dim` sets the hidden width, default `64`. Softmax runs
across the four branches independently for each word. Valid weights sum to one;
invalid word weights and fused vectors are zero. Fusion retains the original
word axis `[B, N, D]` for existing recording pooling and classification.

All four branches, the gate, adapters, and learned acoustic pooling remain
connected to classification gradients. The same decomposition regularizers and
loss coefficients apply to all three decomposed modes. There are no target
gate weights or rules that prescribe weight changes with quality.

For a hidden width `H` and quality width `Q`, a learned gate has
`H * (4 * D + Q + 5) + 4` parameters. At `D=256, H=64`, conventional gating
adds 65,860 parameters and reliability-aware gating adds 66,756. The difference
is 896 parameters. Fixed fusion adds none. Report these counts alongside later
matched experiments.

## Inputs, masks, and checkpoints

Both learned modes require the frame-package input path and the shared word
axis introduced in Steps 3–5. Conventional gating can use pending or ready
quality schemas and does not read quality features when computing its weights.
For matched experiments, use the same ready packages for all three decomposed
modes.

Reliability-aware gating requires the versioned ready schema and the saved
training-fold normalization state from [Step 6](fusion-step-6-reliability-features.md).
Training fits that state automatically on its selected training fold. Evaluation
reuses it; missing or incompatible state and mismatched input fingerprints fail
explicitly. Each valid word needs 14 finite normalized quality values/flags with
the matching feature order, device, and dtype. Missing measurements are already
imputed by Step 6, with their availability flags preserved.

The paired-word mask controls every mode. Padded or unsupported word rows cannot
serve as cross-attention keys or contribute to recording pooling or summaries.
The gate clears invalid quality rows before use, so even nonfinite padding
cannot contaminate predictions or gradients. Each recording still needs at
least one valid pair. Four-way softmax does not introduce missing-modality
support; all valid words have the same four available branches.

Checkpoint settings retain the fusion method and gate hidden width alongside
branch configuration, quality schema, normalization state, and auxiliary loss
weights. Learned gate parameters are registered in the model state dictionary
and included in the existing optimizer. Run names include the mode, branch
width, head count, and learned gate width. The evaluator reconstructs the
architecture and loads strictly. Old fixed-fusion checkpoints need no new
parameters; historical checkpoints without a fusion setting still use legacy
fusion. Changing modes is not an automatic checkpoint migration.

## Enable a learned gate

After preparing ready packages for every selected recording, use the existing
training command with:

```text
--precomputed_input_type frames
--precomputed_features_dir /path/to/reliability_fusion_quality_v1/train
--acoustic_word_pooling attention
--fusion_method reliability_gated
--fusion_gate_hidden_dim 64
--decomposition_dim 256
--decomposition_heads 4
--decomposition_cross_radius 1
```

Use `--fusion_method decomposed_gated` for the conventional learned control,
or `decomposed_fixed` for the fixed control. Mean acoustic pooling is also
supported. Keep pooling, branches, regularization, folds, seeds, and training
budget matched when comparing fusion modes. Step 7 exercised the real pilot;
[Step 8](fusion-step-8-cohort-comparison.md) now prepares the complete development
cohort and the matched comparison.

## Inspect weights

Default forward calls still return logits. For lightweight inspection without
requesting all branches and cross-attention maps:

```python
output = model(frame_batch=inputs, return_fusion_weights=True)
logits = output["logits"]
weights = output["fusion_weights"]       # [B, N, 4], detached
mask = output["fusion_mask"]             # [B, N]
summary = output["fusion_summary"]
branch_order = summary["branch_order"]
mean_weights = summary["mean_weights"]   # [B, 4], valid words only
mean_entropy = summary["mean_entropy"]   # [B], natural-log entropy
valid_words = summary["valid_words"]    # [B]
```

All inspection weights and summaries above are detached and no history is
cached on the model. Summaries average within each recording, so a later cohort
analysis can give recordings equal weight. UIDs and original word IDs remain
available in the input batch for joining results.

`return_representations=True` retains the Step 5 branch inspection interface,
including differentiable `fusion_weights` and detached cross-attention maps.
Custom training loops request `return_auxiliary=True` and add the auxiliary
terms; the stock trainer already does this for every decomposed mode. Weight
inspection alone does not request auxiliary computations.

Weights express relative allocation among branches. They are not calibrated
reliability probabilities or a complete explanation of a clinical prediction.
Quality-input sensitivity confirms wiring; learned robustness requires the
held-out experiments below.

## Verification and remaining work

Ten focused tests cover convex combinations, quality-input isolation,
classification gradients, nonfinite padding, individual/batched inference,
per-recording summaries, required inputs, checkpoint reload and normalization
identity, and actual CLI training/evaluation for both learned modes.

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_fusion_gates.py' -v
```

The combined CPU regression run passes 186 tests, excluding only the existing
ignored import-broken `test_batch_adresso_words.py` documented in Step 3.

The [pilot probe](audits/step7-adrso010-fusion-gate-probe.json) exercises both
learned modes on 1,139 acoustic frames and 41 words, producing four
`[1, 41, 256]` branches and `[1, 41, 4]` weights. It checks group gradients and
updates after one optimizer step, valid-weight normalization, padding,
checkpoint tensors, logits, and weights. Changing quality inputs while holding
embeddings fixed leaves branch tensors unchanged and affects only the
reliability-aware gate's weights. The ready package remains unchanged.
The maximum pilot weight-sum error is approximately `1.2e-7`; reload and
padding logit differences are below `1.8e-7`. Checkpoint tensors match exactly;
tests allow ordinary float32 rounding between attention inference paths.

This is a CPU mechanics probe with fresh models, a synthetic target, and a
one-record normalization fixture, not cohort fold statistics or an efficacy
comparison. No full-cohort performance or GPU results are claimed.

Two planned experimental stages remain:

1. **Step 8:** prepare the remaining cohort packages and compare fixed,
   conventional, and reliability-aware fusion under matched training settings.
2. **Step 9:** evaluate controlled audio noise, transcript corruption, and
   boundary perturbations, reporting classification and per-recording gate
   responses with subgroup checks.
