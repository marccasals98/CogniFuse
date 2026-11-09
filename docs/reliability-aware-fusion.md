# Reliability-aware fusion gate

**Status: proposed design.** This document describes the intended extension to
CogniFuse. The current classifier does not implement this four-branch gate or
consume reliability features. The code below illustrates the proposed operation.

The [Step 1 pilot audit](fusion-step-1-audit.md) verifies saved frame-level inputs.
The [frame-based input contract](fusion-input-contract.md) specifies how to retain
those inputs so word-level acoustic pooling can be learned during training.

## Purpose

For each aligned word or segment, the gate answers:

> How much should the model trust the acoustic information, the textual
> information, and their cross-modal interaction for this particular word?

Reliability is sample-dependent and word-dependent. A recording can contain
both reliable and unreliable words, so fixed modality weights or one global
audio-quality score cannot express all the desired behavior.

The gate should distinguish an unreliable modality from an unreliable pairing.
For example, a confident transcript and usable audio can still have uncertain
word boundaries. In that case, the desired response is to reduce reliance on
their explicit correspondence while retaining useful unimodal information.

## Inputs and representations

For original word index $i$, assume four representations of equal dimension $D$:

| Symbol | Representation | Intended meaning |
|---|---|---|
| $p_i^a$ | Audio-private | Information unique to the acoustic modality |
| $p_i^t$ | Text-private | Information unique to the transcript |
| $s_i$ | Shared audio-text | Information common to both modalities |
| $C_i$ | Cross-modal interaction | Information obtained through cross-attention |

All four tensors must have shape `[B, N, D]`, where `B` is batch size and `N`
is the padded number of original words. Project branches to a common dimension
before taking their weighted sum, and control their scales so gate weights can
be compared meaningfully.

These are intended roles, not properties guaranteed by naming a projection
"private" or "shared". Their construction and any disentanglement or agreement
objectives must be specified when implementing the representation stage.
Likewise, cross-attention can use context from other words; its query indexing,
direction, and attention scope must be defined before interpreting $C_i$ as
evidence about a particular word's correspondence.

## Reliability features

Separate word-level measurement quality from recording-level measurement
quality:

$$
q_i^{\mathrm{token}} =
[q_i^{\mathrm{ASR}}, q_i^{\mathrm{CTC}}, q_i^{\mathrm{align}}]
$$

$$
q^{\mathrm{recording}} =
[q^{\mathrm{SNR}}, q^{\mathrm{clip}}, q^{\mathrm{noise/reverb}}]
$$

$$
q_i = [q_i^{\mathrm{token}}; q^{\mathrm{recording}}].
$$

The recording features are broadcast to every word in that recording. The
three recording entries above are an initial grouping; the final feature count
depends on the chosen noise and reverberation estimators.

| Signal | Candidate source or definition | Interpretation and current availability |
|---|---|---|
| ASR confidence | Whisper word CSV `probability` | Available when exported; a recognition-confidence proxy, not a calibrated probability of correctness |
| CTC confidence | Aggregate Step 4 `mean_log_emission` scores over a word's nonblank character frames, then exponentiate | Character scores and spans exist; word aggregation must be added. This measures emission support for the supplied transcript, not boundary certainty |
| Alignment uncertainty | Entropy of a defined word-boundary posterior | Not currently exported; requires a posterior estimator and an explicit entropy definition |
| Boundary disagreement | Step 6 `delta_start` and `delta_end`, in seconds | Available in comparison reports; can extend or initially substitute for the alignment-uncertainty input, with that choice recorded explicitly |
| SNR proxy | Estimated speech-to-noise ratio | Proposed recording-level extraction; estimator and units must be fixed |
| Clipping ratio | Fraction of source waveform samples near the clipping threshold | Proposed extraction; record threshold and waveform scaling before normalization |
| Noise/reverberation proxy | A specified recording-quality estimator | Proposed extraction; define estimator, range, and direction |

A concrete candidate for word CTC confidence is:

$$
q_i^{\mathrm{CTC}} = \exp\left(
\frac{\sum_{u\in U_i} n_u\ell_u}{\sum_{u\in U_i} n_u}
\right),
$$

where $U_i$ contains the word's CTC character spans, $\ell_u$ is a span's saved
`mean_log_emission`, and $n_u$ is its nonblank frame count
(`end_frame_exclusive - start_frame`). Exclude inter-word delimiters. Words
without aligned units have missing confidence, not confidence zero.

CTC emission entropy, boundary-posterior entropy, and Whisper–CTC boundary
disagreement describe different quantities. Do not relabel one as another.
Disagreement alone does not establish which alignment is correct. If a boundary
posterior $P_i(b)$ is introduced, its entropy can be defined as
$H_i=-\sum_b P_i(b)\log P_i(b)$; document the boundary support and any
normalization needed to compare words.

### Preserve potential clinical information

Do not include pause duration, speech rate, articulation rate, jitter, or shimmer
in the reliability vector. These may be clinical biomarkers. Here, reliability
describes whether a measurement is trustworthy, rather than whether the person's
speech is unusual. Such features can belong in a separate predictive branch.

Even ASR and CTC scores can vary with speech characteristics. Check gate behavior
and predictive performance across diagnostic groups so useful atypical speech
is not systematically suppressed.

### Feature preparation and missing values

- Join features by recording UID and original `word_id`, preserving repeated
  words and row order. Never join by word text alone.
- Save feature names, order, units, estimator versions, source identities, and
  missingness alongside the feature tensors.
- Fit scaling and imputation on each training fold only; reuse them unchanged
  for validation and test data. Quality inputs can have opposite directions:
  high confidence is favorable, while high entropy or clipping is unfavorable.
- Represent missing quality values with finite imputations and explicit
  availability indicators. Missing quality is distinct from an invalid embedding.
- Exclude padding and invalid word pairs from attention, pooling, and gate
  summaries. Quality values do not override validity masks.

## Four-way gate

Concatenate the representations and reliability features:

$$
h_i = [p_i^a; p_i^t; s_i; C_i; q_i].
$$

A small learned gate produces four logits and normalizes them across branches:

$$
[g_i^a,g_i^t,g_i^s,g_i^{\mathrm{cross}}]
= \operatorname{softmax}(W_g h_i+b_g).
$$

A small MLP can replace the affine map. Its output dimension remains four,
and softmax runs over branches, not words. For each valid word:

$$
g_i^a+g_i^t+g_i^s+g_i^{\mathrm{cross}}=1,
\qquad g_i^k\geq0.
$$

The fused word representation is:

$$
z_i=g_i^a p_i^a+g_i^t p_i^t+g_i^s s_i+g_i^{\mathrm{cross}} C_i.
$$

The result has shape `[B, N, D]` and can feed masked recording-level pooling
and the classifier. The weights express relative branch allocation; they are
not calibrated reliability probabilities or a complete explanation of the
prediction. All four inputs may be poor even though the weights sum to one.

### Expected behavior

| Situation | Desired tendency to test |
|---|---|
| Clean audio, high ASR confidence, clear alignment | Greater shared and/or cross-modal contribution |
| Noisy audio, usable transcript | Lower audio-private weight; relatively higher text-private weight |
| Clean audio, poor transcript | Lower text-private weight; relatively higher audio-private weight |
| Whisper and CTC disagree about boundaries | Lower cross-modal weight; unimodal information can remain useful |
| Both modalities are confident and agree | Strong shared and/or cross-modal contribution |

For boundary disagreement, $g_i^{\mathrm{cross}}$ should be able to decrease
without requiring both $g_i^a$ and $g_i^t$ to decrease. Their combined relative
share can increase. However, if the audio word vector was pooled from incorrect
frames, the audio-private branch may also be unreliable; the gate cannot recover
acoustic evidence that preprocessing assigned to the wrong word.

These are hypotheses to evaluate, not manually assigned rules. An unconstrained
learned gate does not guarantee monotonic responses to the quality features.

## Learning and minimal computation

Train the gate through the recording-level classification objective. For a
binary task, an illustrative loss is:

$$
\mathcal L_{\mathrm{cls}} =
-[y\log\hat y+(1-y)\log(1-\hat y)].
$$

Use the experiment's configured loss for multiclass tasks. Gradients flow from
the classifier through pooling and fusion into the gate and trainable branches.
The initial design needs no target gate weights or hand-coded quality thresholds.

Illustrative PyTorch computation, assuming valid, finite inputs and an already
constructed `q` (including any missingness indicators):

```python
import torch
from torch import nn

# Four branch tensors: [B, N, D]; q: [B, N, Q].
# Define once in the model's __init__ so parameters are registered and learned.
gate_mlp = nn.Linear(4 * D + Q, 4)

# In forward:
gate_input = torch.cat(
    [audio_private, text_private, shared, cross_modal, q], dim=-1
)
weights = torch.softmax(gate_mlp(gate_input), dim=-1)  # [B, N, 4]
z = (
    weights[..., 0:1] * audio_private
    + weights[..., 1:2] * text_private
    + weights[..., 2:3] * shared
    + weights[..., 3:4] * cross_modal
)
z = z.masked_fill(~word_mask.unsqueeze(-1), 0)
# Pool with word_mask; exclude masked positions when summarizing weights.
```

## Integration with the current repository

The current path in [`scripts/model.py`](../scripts/model.py) normalizes audio
and text features, applies modality adapters, then runs sequence-to-sequence
attention, sequence-to-one pooling, and classification. It does not construct
the proposed private/shared decomposition. Existing cross-attention options in
[`scripts/poolings.py`](../scripts/poolings.py) also need an explicit mapping
to one interaction vector per original word before use in this gate.

The precomputed word workflow is a suitable first integration target:

1. Extract and version reliability features from the existing Whisper CSVs,
   [`CTC alignment`](../utils/ctc_forced_alignment.py),
   [`alignment comparison`](../utils/compare_word_alignment.py), and waveform.
   Choose disagreement features or implement boundary entropy explicitly.
2. Extend [`packaging`](../utils/package_word_embeddings.py) and
   [`data loading`](../scripts/data.py) to carry aligned quality tensors and
   availability indicators through padding and batching.
3. Implement the four representation branches with a shared word axis and
   dimension, then insert the gate before masked recording pooling. The fused
   sequence has `N` word rows; update the output mask rather than reusing a
   concatenated audio/text mask of length `2N`.
4. Make gate configuration and quality preprocessing reproducible in checkpoints,
   and expose valid-word gate summaries for evaluation.
5. Run the ablations and degradation experiments below before drawing conclusions
   about reliability awareness.

Current word packaging retains all original rows but uses the audio/text mask
intersection for both training masks. Initially preserve that policy. Supporting
a missing modality at a word requires a separate branch-availability design,
masked gate logits, and an explicit policy for words with no available branch.
A softmax over four branches alone does not provide missing-modality support.

## Validation and ablations

The essential comparison uses the same representations and training protocol:

| Variant | Gate input | Question |
|---|---|---|
| Conventional learned gate | $[p_i^a;p_i^t;s_i;C_i]$ | What can representation-dependent gating achieve? |
| Reliability-aware gate | $[p_i^a;p_i^t;s_i;C_i;q_i]$ | Does explicit measurement-quality information improve fusion? |

Keep branch construction, folds, seeds, loss, and training budget matched, and
report gate parameter counts. Optional controls include fixed equal weights,
token-only quality, recording-only quality, and shuffled quality inputs.

Evaluate controlled degradations of held-out recordings with a fixed trained
model and clean references:

| Intervention | Example severity levels | Primary gate response to test |
|---|---|---|
| Add audio noise | SNR 30, 20, 10, 5 dB | Decreasing $g^a$ |
| Corrupt transcript words | WER 0%, 10%, 20%, 30% | Decreasing $g^t$ |
| Perturb word boundaries | 0, ±100, ±200, ±500 ms | Decreasing $g^{\mathrm{cross}}$ |

Record achieved WER and the exact corruption procedure; substitutions are a
simple initial intervention that preserves word indices. Boundary perturbations
must keep intervals valid and within the recording. Track any lost valid pairs
so changes in gate averages are not caused only by changing the evaluated words.

Recompute all affected inputs: noisy waveforms change acoustic embeddings and
audio quality; corrupted text changes text embeddings and relevant confidence
signals; changed boundaries require re-pooling acoustic frames. Specify whether
the transcript and alignments are fixed to isolate one effect or regenerated for
an end-to-end experiment. These protocols answer different questions. Separately,
varying only $q_i$ with embeddings fixed probes gate sensitivity to quality inputs.

Report classification performance alongside per-branch weight distributions and
paired changes from clean inputs. Aggregate within recordings before cohort
summaries, show variability across recordings and folds, and inspect diagnostic
subgroups. Preserve the existing patient-disjoint split policy where applicable,
and keep test data out of feature normalization and model selection.

The intended claim is **adaptive modality reliance under controlled degradation**.
Support it with both gate-response evidence and predictive robustness; learned
weights alone do not establish that claim.
