"""Regularized word branches with fixed or learned four-way fusion."""

import math

import torch
from torch import nn
from torch.nn import functional as F

DECOMPOSED_METHODS = ('decomposed_fixed', 'decomposed_gated', 'reliability_gated')
FUSION_METHODS = ('legacy', *DECOMPOSED_METHODS)
BRANCH_NAMES = ('audio_private', 'text_private', 'shared', 'cross_modal')


LOSS_WEIGHTS = {
    'shared_agreement': ('shared_agreement_weight', 0.1),
    'private_shared_separation': ('private_shared_separation_weight', 0.01),
    'reconstruction': ('decomposition_reconstruction_weight', 0.1),
    'shared_variance': ('shared_variance_weight', 0.1),
}


def validate_fusion_configuration(parameters):
    method = getattr(parameters, 'fusion_method', 'legacy')
    if method not in FUSION_METHODS:
        raise ValueError('Unknown fusion method')
    if method == 'legacy':
        return
    if (getattr(parameters, 'precomputed_input_type', 'words') != 'frames'
            or not getattr(parameters, 'precomputed_features_dir', None)):
        raise ValueError('Decomposed word fusion requires precomputed frame packages')
    dim = getattr(parameters, 'decomposition_dim', 256)
    heads = getattr(parameters, 'decomposition_heads', 4)
    radius = getattr(parameters, 'decomposition_cross_radius', 1)
    if type(dim) is not int or dim < 3 or type(heads) is not int or heads < 1 or dim % heads:
        raise ValueError('Decomposition dimension must be >=3 and divisible by positive head count')
    if type(radius) is not int or radius == 0 or radius < -1:
        raise ValueError('Cross-attention radius must be positive, or -1 for the full recording')
    hidden = getattr(parameters, 'fusion_gate_hidden_dim', 64)
    if type(hidden) is not int or hidden < 1:
        raise ValueError('Fusion gate hidden dimension must be a positive integer')
    for name, default in LOSS_WEIGHTS.values():
        weight = getattr(parameters, name, default)
        if not isinstance(weight, (float, int)) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f'{name} must be finite and nonnegative')


def per_recording_mean(values, mask):
    """Give recordings equal weight regardless of their valid word counts."""
    return ((values.masked_fill(~mask, 0).sum(dim=1)) / mask.sum(dim=1)).mean()


class DecomposedWordFusion(nn.Module):
    """One representation and four normalized branch weights per valid word."""

    def __init__(self, input_dim, dim=256, heads=4, cross_radius=1,
                 fusion_method='decomposed_fixed', gate_hidden_dim=64, quality_dim=0):
        super().__init__()
        if (type(input_dim) is not int or input_dim < 1 or type(dim) is not int or dim < 3
                or type(heads) is not int or heads < 1 or dim % heads):
            raise ValueError('Invalid decomposition dimensions or head count')
        if type(cross_radius) is not int or cross_radius == 0 or cross_radius < -1:
            raise ValueError('Cross-attention radius must be positive, or -1 for the full recording')
        if fusion_method not in DECOMPOSED_METHODS:
            raise ValueError('Unknown decomposed fusion method')
        if type(gate_hidden_dim) is not int or gate_hidden_dim < 1:
            raise ValueError('Fusion gate hidden dimension must be a positive integer')
        if type(quality_dim) is not int or quality_dim < 0 or (
                (fusion_method == 'reliability_gated') != (quality_dim > 0)):
            raise ValueError('Only reliability_gated fusion requires a positive quality dimension')
        self.input_dim, self.dim, self.heads, self.cross_radius = input_dim, dim, heads, cross_radius
        self.fusion_method, self.quality_dim = fusion_method, quality_dim

        def encoder():
            return nn.Sequential(nn.Linear(input_dim, dim), nn.Tanh(), nn.LayerNorm(dim))

        self.audio_private = encoder()
        self.text_private = encoder()
        self.shared_encoder = encoder()  # Exactly the same parameters for both modalities.
        self.shared_norm = nn.LayerNorm(dim)
        self.audio_queries_text = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.text_queries_audio = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.cross_merge = nn.Sequential(nn.Linear(2 * dim, dim), nn.LayerNorm(dim))
        self.audio_decoder = nn.Linear(2 * dim, input_dim)
        self.text_decoder = nn.Linear(2 * dim, input_dim)
        # Fixed fusion has no new parameters, preserving Step 5 checkpoints.
        if fusion_method != 'decomposed_fixed':
            # Gate width must not change subsequent common-layer initialization
            # or the training RNG stream in matched fusion comparisons.
            with torch.random.fork_rng(devices=[]):
                self.gate = nn.Sequential(nn.Linear(4 * dim + quality_dim, gate_hidden_dim),
                                          nn.Tanh(), nn.Linear(gate_hidden_dim, 4))

    def attention_mask(self, mask):
        n = mask.shape[1]
        positions = torch.arange(n, device=mask.device)
        local = (positions[:, None] - positions[None, :]).abs() <= self.cross_radius
        if self.cross_radius == -1:
            local = torch.ones_like(local)
        allowed = local[None] & mask[:, None, :]
        # Padded queries use the recording's valid keys before being zeroed.
        # This prevents all-masked softmax rows and NaN backward passes.
        allowed = torch.where(mask[:, :, None], allowed, mask[:, None, :])
        return (~allowed).repeat_interleave(self.heads, dim=0)

    def forward(self, audio, text, mask, return_auxiliary=False, return_attention=False,
                reliability_features=None):
        if (audio.ndim != 3 or audio.shape != text.shape or audio.shape[-1] != self.input_dim
                or mask.dtype != torch.bool or tuple(mask.shape) != tuple(audio.shape[:2])):
            raise ValueError('Decomposition requires paired [B,N,input_dim] features and a boolean word mask')
        if not bool(mask.any(dim=1).all()):
            raise ValueError('Every recording requires at least one valid word pair')

        def clear(value):
            return value.masked_fill(~mask[..., None], 0)

        if self.quality_dim:
            if (reliability_features is None or reliability_features.shape != (*mask.shape, self.quality_dim)
                    or reliability_features.device != audio.device or reliability_features.dtype != audio.dtype):
                raise ValueError('Reliability fusion requires matching [B,N,quality_dim] features, device, and dtype')
            reliability_features = clear(reliability_features)
            if not bool(torch.isfinite(reliability_features).all()):
                raise ValueError('Reliability features must be finite on valid words')
        audio, text = clear(audio), clear(text)
        pa, pt = clear(self.audio_private(audio)), clear(self.text_private(text))
        sa, st = clear(self.shared_encoder(audio)), clear(self.shared_encoder(text))
        shared = clear(self.shared_norm((sa + st) / 2))
        ua = clear(F.layer_norm(pa + sa, (self.dim,)))
        ut = clear(F.layer_norm(pt + st, (self.dim,)))
        attention_mask = self.attention_mask(mask)
        a_context, a_weights = self.audio_queries_text(
            ua, ut, ut, attn_mask=attention_mask, need_weights=return_attention,
            average_attn_weights=False)
        t_context, t_weights = self.text_queries_audio(
            ut, ua, ua, attn_mask=attention_mask, need_weights=return_attention,
            average_attn_weights=False)
        cross = clear(self.cross_merge(torch.cat((a_context, t_context), dim=-1)))
        if self.fusion_method == 'decomposed_fixed':
            weights = mask[..., None].to(audio.dtype).expand(-1, -1, 4) / 4
            fused = (pa + pt + shared + cross) / 4
        else:
            gate_inputs = [pa, pt, shared, cross]
            if self.quality_dim:
                gate_inputs.append(reliability_features)
            weights = clear(torch.softmax(self.gate(torch.cat(gate_inputs, dim=-1)), dim=-1))
            fused = sum(weights[..., j:j+1] * branch for j, branch in enumerate((pa, pt, shared, cross)))
        result = {'audio_private': pa, 'text_private': pt, 'shared_audio': sa,
                  'shared_text': st, 'shared': shared, 'cross_modal': cross,
                  'fused': fused, 'mask': mask,
                  'fusion_weights': weights,
                  'auxiliary_losses': {}}
        if return_auxiliary:
            reconstructed_audio = clear(self.audio_decoder(torch.cat((pa, sa), dim=-1)))
            reconstructed_text = clear(self.text_decoder(torch.cat((pt, st), dim=-1)))
            agreement = (sa - st).square().mean(dim=-1)
            separation = (F.cosine_similarity(pa, sa, dim=-1).square()
                          + F.cosine_similarity(pt, st, dim=-1).square()) / 2
            reconstruction = ((reconstructed_audio - audio.detach()).square().mean(dim=-1)
                              + (reconstructed_text - text.detach()).square().mean(dim=-1)) / 2
            variance_terms = []
            for b in range(len(mask)):
                if int(mask[b].sum()) < 2:
                    variance_terms.append((sa[b].sum() + st[b].sum()) * 0)
                else:
                    std_a = (sa[b, mask[b]].var(dim=0, unbiased=False) + 1e-4).sqrt()
                    std_t = (st[b, mask[b]].var(dim=0, unbiased=False) + 1e-4).sqrt()
                    variance_terms.append((F.relu(1 - std_a).mean() + F.relu(1 - std_t).mean()) / 2)
            result['auxiliary_losses'] = {
                'shared_agreement': per_recording_mean(agreement, mask),
                'private_shared_separation': per_recording_mean(separation, mask),
                'reconstruction': per_recording_mean(reconstruction, mask),
                'shared_variance': torch.stack(variance_terms).mean(),
            }
            result.update(reconstructed_audio=reconstructed_audio, reconstructed_text=reconstructed_text)
        if return_attention:
            query_mask = mask[:, None, :, None]
            result['cross_attention'] = {
                'audio_queries_text': a_weights.detach().masked_fill(~query_mask, 0),
                'text_queries_audio': t_weights.detach().masked_fill(~query_mask, 0),
            }
        return result


def summarize_fusion_weights(weights, mask):
    """Detached per-recording summaries; invalid words never enter averages."""
    with torch.no_grad():
        clean = weights.detach().masked_fill(~mask[..., None], 0)
        counts = mask.sum(dim=1)
        mean = clean.sum(dim=1) / counts[:, None]
        entropy = -(clean * clean.clamp_min(torch.finfo(clean.dtype).tiny).log()).sum(dim=-1)
        return {'branch_order': BRANCH_NAMES, 'valid_words': counts,
                'mean_weights': mean, 'mean_entropy': entropy.sum(dim=1) / counts}
