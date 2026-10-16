"""ADReSSo diagnosis recording index and saved development partitions.

Recording UIDs are used as grouping IDs because this distribution has no
independent participant-ID table. Test recordings have no diagnostic label.
This module reads paths and split files; it never loads or modifies features.
"""

from collections import Counter
from pathlib import Path

import numpy as np


ADRESSO_LABELS = {"cn": 0, "ad": 1}
UNLABELED = -1


class ADReSSoIndex:
    def __init__(self, dataset_root):
        if not dataset_root:
            raise ValueError("ADReSSo requires --dataset_root")
        root = Path(dataset_root).expanduser().resolve()
        self.root = root / "diagnosis" if (root / "diagnosis").is_dir() else root
        self.training = self._recordings("train")
        self.test = self._recordings("test-dist")
        if {r["uid"] for r in self.training} & {r["uid"] for r in self.test}:
            raise ValueError("ADReSSo train/test recording UIDs overlap")

    def _recordings(self, partition):
        audio_dir = self.root / partition / "audio"
        if not audio_dir.is_dir():
            raise FileNotFoundError(f"Missing ADReSSo audio directory: {audio_dir}")
        records, seen = [], set()
        for path in sorted(audio_dir.rglob("*.wav")):
            uid = path.stem
            if uid in seen:
                raise ValueError(f"Duplicate ADReSSo recording UID: {uid}")
            seen.add(uid)
            diagnosis = None
            if partition == "train":
                relative = path.relative_to(audio_dir)
                if len(relative.parts) != 2 or relative.parts[0] not in ADRESSO_LABELS:
                    raise ValueError(f"Training audio must be under audio/ad or audio/cn: {path}")
                diagnosis = relative.parts[0]
            elif path.parent != audio_dir:
                raise ValueError(f"Test audio must be directly under test-dist/audio: {path}")
            records.append({"uid": uid, "filename": str(path.relative_to(audio_dir)),
                            "audio_path": str(path), "patient_id": uid,
                            "diagnosis": diagnosis,
                            "label": ADRESSO_LABELS[diagnosis] if diagnosis else UNLABELED,
                            "partition": partition})
        if not records:
            raise ValueError(f"No ADReSSo WAV recordings found: {audio_dir}")
        return records

    @staticmethod
    def _read_uids(path):
        values = np.load(path, allow_pickle=False)
        if values.ndim != 1 or values.dtype.kind not in {"U", "S"}:
            raise ValueError(f"Saved split must be a one-dimensional string array: {path}")
        uids = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values.tolist()]
        if not uids or any(not uid or uid != uid.strip() for uid in uids):
            raise ValueError(f"Saved split contains empty or invalid UIDs: {path}")
        if len(uids) != len(set(uids)):
            raise ValueError(f"Duplicate UIDs in saved split: {path}")
        return uids

    def development_folds(self, num_folds=5):
        if num_folds < 2:
            raise ValueError("num_folds must be at least 2")
        expected = {r["uid"] for r in self.training}
        test = {r["uid"] for r in self.test}
        folds, held_out = [], set()
        split_dir = self.root / "train" / "splits"
        expected_files = {f"{split}_uids{fold}.npy" for split in ("train", "val") for fold in range(num_folds)}
        existing_files = {p.name for pattern in ("train_uids*.npy", "val_uids*.npy") for p in split_dir.glob(pattern)}
        if existing_files != expected_files:
            raise ValueError("num_folds must match the complete saved ADReSSo split files; "
                             f"missing={sorted(expected_files - existing_files)}, "
                             f"unexpected={sorted(existing_files - expected_files)}")
        for fold in range(num_folds):
            train = self._read_uids(split_dir / f"train_uids{fold}.npy")
            val = self._read_uids(split_dir / f"val_uids{fold}.npy")
            a, b = set(train), set(val)
            if (a | b) & test:
                raise ValueError(f"Official test UID included in development fold {fold}")
            if a & b:
                raise ValueError(f"Training/validation overlap in saved fold {fold}")
            if a | b != expected:
                raise ValueError(f"Saved fold {fold} does not partition the complete training cohort; "
                                 f"missing={sorted(expected - (a | b))}, unknown={sorted((a | b) - expected)}")
            if b & held_out:
                raise ValueError("A recording appears in validation in multiple saved folds")
            held_out.update(b)
            folds.append({"fold": fold, "training_uids": train, "validation_uids": val})
        if held_out != expected:
            raise ValueError("Saved folds do not hold out each training recording once")
        return folds

    def select(self, split, fold=None, num_folds=5):
        split = split.lower()
        if split == "test":
            return list(self.test)
        if split not in {"train", "val", "validation"}:
            raise ValueError("split must be train, val, validation, or test")
        if fold is None or not 0 <= fold < num_folds:
            raise ValueError(f"ADReSSo development requires an explicit fold in 0..{num_folds - 1}")
        plan = self.development_folds(num_folds)[fold]
        key = "training_uids" if split == "train" else "validation_uids"
        by_uid = {r["uid"]: r for r in self.training}
        return [by_uid[uid] for uid in plan[key]]

    def summary(self, num_folds=5):
        labels = {r["uid"]: r["diagnosis"] for r in self.training}
        folds = self.development_folds(num_folds)
        return {"dataset": "adresso", "diagnosis_root": str(self.root),
                "label_map": ADRESSO_LABELS, "training_recordings": len(self.training),
                "training_classes": dict(Counter(labels.values())),
                "test_recordings": len(self.test), "test_labels_available": False,
                "group_id_source": "recording_uid; independent participant identity unverified",
                "folds": [{"fold": f["fold"],
                           "training_recordings": len(f["training_uids"]),
                           "validation_recordings": len(f["validation_uids"]),
                           "validation_classes": dict(Counter(labels[uid] for uid in f["validation_uids"]))}
                          for f in folds]}
