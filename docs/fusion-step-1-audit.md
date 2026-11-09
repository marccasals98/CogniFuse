# Step 1: pilot input audit

**Completed on November 9, 2026.** Audited `train/adrso010` from
`outputs/adresso_deberta_v3_large_wavlm_large_ctc_20261105/`.
The recording is in the development partition; no test predictions or model
selection were performed. The latest encoder variant was selected because its
complete intermediates remain available.

The [machine-readable report](audits/adrso010-wavlm-deberta.json) includes exact
source/stage paths, checkpoint revisions, and hashes for 45 verified files.
The [input contract](fusion-input-contract.md) defines the proposed package
interface for Step 2.

## What is available

| Artifact | Audited result | Reuse decision |
|---|---|---|
| Audio | 22.7853 s after resampling to 16 kHz; original WAV remains available | Reuse source for later waveform-quality extraction |
| Acoustic encoder | `microsoft/wavlm-large`, revision `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c` | Reuse frozen `last_hidden_state` |
| Acoustic frames | float32 `[1139, 1024]`; all frames valid; 20 ms stride, 25 ms convolution support | Reuse without rerunning the encoder |
| CTC aligner | `facebook/wav2vec2-base-960h`, revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156` | Reuse character alignment and word windows |
| Original words | 41 rows, IDs `0..40`; four word types occur more than once | Preserve original IDs and row order |
| Acoustic word windows | 41 valid windows; zero added context; 1–41 frames per word, median 11 | Reuse as the initial attention windows |
| Text encoder | `microsoft/deberta-v3-large`, revision `64a8c8eab3e352a784c658aef62be1662607476f` | Reuse frozen word vectors |
| Text vectors | float32 `[41, 1024]`; one encoder chunk, 200-token limit; zero truncated tokens | Reuse without rerunning the text encoder |
| Masks | 41 valid audio words, 41 valid text words, 41 valid pairs | Preserve initial paired-validity policy |
| Whisper confidence | Available for all 41 words | Read original CSV `probability` by word ID |
| CTC emission confidence | Derivable for all 41 words from saved character scores/spans | Add versioned word aggregation; no encoder inference needed |
| Boundary disagreement | Start/end differences available for all 41 words | Reuse unexpanded Whisper/CTC comparison |
| Recording-quality vector | Not exported | Define and extract estimators in the quality stage |
| Boundary entropy | Not exported; no posterior definition | Defer; use clearly named disagreement features initially |

## Checks performed

- Verified original audio/word CSV and published-package hashes against the
  batch summary, and all eight stage completion manifests against their files.
- Verified that stage signatures use the same source hashes and extraction
  configuration, including the actual acoustic/text/CTC checkpoint revisions.
- Validated the published package against its Step 7/8 source vectors and masks.
- Recomputed timing-aware acoustic word mapping and raw-frame mean pooling.
  **Maximum absolute error against the saved acoustic word vectors: `0.0`.**
- Recomputed pooling metadata and the Whisper/CTC comparison; both match their
  saved versions. Checked original word IDs, text mappings, and mask consistency.
- Checked character score validity and derived frame-weighted word CTC emission
  confidence, excluding inter-word delimiters.

The CTC and WavLM convolution grids happen to match for all 41 pilot words.
The data contract still requires timing-based mapping for other encoder pairs.

A compatibility probe of the older DistilBERT/Wav2Vec2 export for the same UID
found that its acoustic frame metadata omits the checkpoint revision. The new
strict audit rejects that omission rather than assuming a revision from the
model name. This does not affect the selected WavLM pilot. Reusing that older
variant would require separately establishing its extraction provenance.

## Interpretation and limits

This pilot can support trainable frame-to-word attention using existing frozen
encoder outputs. No audio/text encoder regeneration is needed for this recording
with the current checkpoints, source transcript, and extraction settings.

Eleven words exceed the existing 200 ms review threshold for at least one
Whisper/CTC boundary. The maximum start disagreement is 720 ms; the maximum end
disagreement is 295 ms. These are measurement disagreements, not proof that
either method is incorrect, and they do not invalidate otherwise usable rows.
Retain the continuous differences as quality signals rather than treating the
review threshold as an automatic rejection rule.

Attention over fixed windows can learn which saved frames to emphasize. It
cannot recover the correct word if its evidence lies entirely outside the
window; explicit context expansion or another alignment strategy would be a
separate experiment. Some windows contain only one frame, so attention has no
weighting choice for those words.

This audit establishes provenance, structural consistency, and exact mean
reconstruction for one recording. It does not validate clinical performance,
ground-truth boundary accuracy, cohort-wide readiness, missing-modality support,
or handling of all invalid/missing-data cases. Those need later checks.

## Reproduce the audit

The audit command reads existing artifacts without running encoders, changing
source features, or training a model. It requires the intermediate files and
original source paths recorded by the existing export, and explicit acoustic
checkpoint revision metadata.

```bash
.venv/bin/python -m utils.audit_fusion_inputs \
  --feature-root outputs/adresso_deberta_v3_large_wavlm_large_ctc_20261105 \
  --uid adrso010
```

Omitting `--output` prints JSON. Pass `--output /tmp/fusion-audit.json` to save a
new report; an existing report is refused. `--partition` defaults to `train`.

The next step is to implement a separate package exporter using the input
contract. The existing model, loaders, feature packages, and experiments have
not been modified by Step 1.
