"""Masks for the token-aligned tensors saved by prepro_embeddings.py."""

import torch


def embedding_masks(attention_mask, offsets):
    """Keep text special tokens, aligned audio tokens, and the audio summary.

    Extraction stores a whole-recording audio mean at position zero. Other
    audio special-token positions (e.g. SEP) have no features and are excluded.
    """
    text_mask = torch.as_tensor(attention_mask, dtype=torch.bool).detach().cpu()
    offsets = torch.as_tensor(offsets).cpu()
    if text_mask.ndim != 1 or offsets.shape != (len(text_mask), 2) or not text_mask.any():
        raise ValueError("Expected a nonempty text mask and one character offset pair per position")
    audio_mask = text_mask & (offsets[:, 0] != offsets[:, 1])
    audio_mask[0] = True
    return audio_mask, text_mask
