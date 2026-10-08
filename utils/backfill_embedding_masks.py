"""Reconstruct masks without rerunning Whisper or either feature encoder.

Use the original transcripts, tokenizer and maximum length used for extraction.
All pairs are validated before any new sidecar is written. Existing matching
masks are kept; conflicting masks cause an error rather than being overwritten.
"""

import argparse
from pathlib import Path
import os
import tempfile
import unicodedata

import pandas as pd
import torch

from .embedding_masks import embedding_masks
from .word_alignment import token_audio_intervals


def prepare_masks(transcriptions, embeddings_dir, tokenizer, max_length=200,
                  text_suffix="distil.pt", audio_suffix="distil_audio.pt",
                  transcript_column="transcription", words_dir=None):
    """Validate the entire migration and return only sidecars that are missing."""
    transcriptions, embeddings_dir = Path(transcriptions), Path(embeddings_dir)
    words_dir = Path(words_dir) if words_dir else transcriptions.parent / "words"
    for suffix in (text_suffix, audio_suffix):
        if Path(suffix).name != suffix or not suffix.endswith(".pt"):
            raise ValueError("Feature suffixes must be filenames ending in .pt")
    if text_suffix == audio_suffix:
        raise ValueError("Audio and text suffixes must differ")
    df = pd.read_csv(transcriptions, dtype={"uid": str}, keep_default_na=False)
    if df.uid.duplicated().any():
        raise ValueError("Transcript UIDs must be unique")
    pending = {}
    for _, row in df.iterrows():
        uid = row["uid"]
        if not uid or Path(uid).name != uid:
            raise ValueError("Transcript UIDs must be nonempty filenames without a directory")
        transcription = unicodedata.normalize("NFC", str(row[transcript_column]))
        encoded = tokenizer(
            transcription, padding="max_length", truncation=True,
            max_length=max_length, return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = encoded["offset_mapping"][0]
        audio_mask, text_mask = embedding_masks(encoded["attention_mask"][0], offsets)
        words = pd.read_csv(words_dir / (uid + ".csv"), dtype={"word": str}, keep_default_na=False)
        token_audio_intervals(
            transcription, list(words[["word", "start", "end"]].itertuples(index=False, name=None)),
            offsets.tolist(),
        )
        for suffix, mask in ((audio_suffix, audio_mask), (text_suffix, text_mask)):
            path = embeddings_dir / (uid + suffix)
            features = torch.load(path, map_location="cpu", weights_only=True)
            if (not isinstance(features, torch.Tensor) or features.ndim != 2
                    or features.shape[0] != max_length or features.shape[1] < 1
                    or not features.is_floating_point() or not torch.isfinite(features).all()):
                raise ValueError(f"Invalid feature tensor or extraction length mismatch: {path}")
            if suffix == audio_suffix:
                # This layout was written by the original extraction code. It
                # provides a consistency check, not a way to infer text masks.
                if (features[~mask] != 0).any() or (features[mask].abs().sum(dim=1) == 0).any():
                    raise ValueError(f"Audio positions disagree with reconstructed mask: {path}")
            mask_path = path.with_name(path.stem + "_mask.pt")
            if mask_path.exists():
                saved = torch.load(mask_path, map_location="cpu", weights_only=True)
                if not isinstance(saved, torch.Tensor) or saved.dtype != torch.bool or not torch.equal(saved, mask):
                    raise ValueError(f"Existing mask conflicts with reconstruction: {mask_path}")
            else:
                pending[mask_path] = mask
    return pending, len(df)


def write_masks(pending):
    for path, mask in pending.items():
        # Publish complete sidecars; never leave a partially serialized .pt file.
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            torch.save(mask, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcriptions", required=True, type=Path)
    parser.add_argument("--embeddings-dir", required=True, type=Path)
    parser.add_argument("--words-dir", type=Path)
    parser.add_argument("--tokenizer", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--text-suffix", default="distil.pt")
    parser.add_argument("--audio-suffix", default="distil_audio.pt")
    parser.add_argument("--transcript-column", default="transcription")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, use_fast=True, local_files_only=args.local_files_only,
    )
    pending, count = prepare_masks(
        args.transcriptions, args.embeddings_dir, tokenizer, args.max_length,
        args.text_suffix, args.audio_suffix, args.transcript_column, args.words_dir,
    )
    if args.dry_run:
        print(f"Validated {count} recording pairs; would create {len(pending)} masks.")
    else:
        write_masks(pending)
        print(f"Validated {count} recording pairs; created {len(pending)} masks.")


if __name__ == "__main__":
    main()
