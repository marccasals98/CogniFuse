import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.data import ADDataset, PrecomputedADDataset


class PrecomputedDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rows = []
        for diagnosis in ("lvPPA", "nfPPA", "svPPA"):
            for patient in range(5):
                uid = f"{diagnosis}_{patient}"
                rows.append({"filename": uid + ".wav", "NHC ID HSP": uid, "DX_Pilar": diagnosis})
                torch.save(torch.randn(6, 4), self.root / f"{uid}distil_audio.pt")
                torch.save(torch.randn(6, 4), self.root / f"{uid}distil.pt")
        self.csv = self.root / "labels.csv"
        pd.DataFrame(rows).to_csv(self.csv, index=False)
        self.params = SimpleNamespace(
            train_labels_path=str(self.csv), validation_labels_path=str(self.csv),
            precomputed_features_dir=str(self.root), num_folds=5, random_seed=1234,
        )

    def dataset(self, **kwargs):
        return PrecomputedADDataset(self.params, fold=1, **kwargs)

    def test_patient_split_matches_raw_dataset_without_audio_or_models(self):
        with patch.object(ADDataset, "__init__", side_effect=AssertionError("raw setup")):
            train, val = self.dataset(), self.dataset(split="val")
        self.assertEqual(len(train), 12)
        self.assertEqual(len(val), 3)
        self.assertFalse(set(train.df.patient_id) & set(val.df.patient_id))
        pd.testing.assert_frame_equal(train.df, ADDataset.create_patient_split(train, train.load_and_prepare_csv()))
        self.assertEqual(train.class_counts, {0: 4, 1: 4, 2: 4})
        self.assertEqual(train.get_classes_weights(), [3., 3., 3.])
        self.assertFalse(list(self.root.glob("trans_*")))

    def test_values_dtype_and_batch(self):
        dataset = self.dataset()
        record = dataset.segments[0]
        saved = torch.full((6, 4), 0.375, dtype=torch.float16)
        torch.save(saved, record["text_path"])
        speech, label, text = dataset[0]
        self.assertEqual(text.dtype, torch.float32)
        torch.testing.assert_close(text, saved.float())
        self.assertEqual(speech.device.type, "cpu")
        self.assertEqual(label.dtype, torch.long)
        batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)))
        self.assertEqual([tuple(x.shape) for x in batch], [(2, 6, 4), (2,), (2, 6, 4), (2, 6)])
        self.assertTrue(batch[3].all())

    def test_missing_pair_fails(self):
        dataset = self.dataset()
        Path(dataset.segments[0]["speech_path"]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing precomputed"):
            self.dataset()

    def test_invalid_tensors_fail(self):
        dataset = self.dataset()
        path = dataset.segments[0]["text_path"]
        for invalid in (torch.ones(6, dtype=torch.long), torch.full((6, 4), float("nan")), {"features": torch.ones(6, 4)}):
            with self.subTest(invalid=type(invalid)):
                torch.save(invalid, path)
                with self.assertRaises(ValueError):
                    dataset[0]
        torch.save(torch.ones(5, 4), path)
        with self.assertRaisesRegex(ValueError, "token counts differ"):
            dataset[0]
        torch.save(torch.ones(6, 8), path)
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            dataset[0]

    def test_uid_collision_across_patients_fails(self):
        rows = pd.read_csv(self.csv)
        duplicate = rows.iloc[[0]].copy()
        duplicate["NHC ID HSP"] = "another_patient"
        pd.concat([rows, duplicate]).to_csv(self.csv, index=False)
        with self.assertRaisesRegex(ValueError, "Multiple recordings"):
            self.dataset()

    def test_custom_suffixes(self):
        for path in self.root.glob("*distil*.pt"):
            path.rename(path.with_name(path.name.replace("distil", "roberta_pauses")))
        self.params.precomputed_text_suffix = "roberta_pauses.pt"
        self.params.precomputed_audio_suffix = "roberta_pauses_audio.pt"
        self.assertEqual(len(self.dataset()), 12)

    def test_training_and_evaluation_without_encoders(self):
        # A separate process preserves train.py's script-style imports without
        # colliding with the repository's utils namespace in other tests.
        code = '''
import sys
from unittest.mock import patch
import torch
sys.path.insert(0, "scripts")
from train import ArgsParser, Trainer
from model import Classifier
parser = ArgsParser()
parser.add_parser_args()
params = parser.parser.parse_args(sys.argv[1:])
with patch.object(Classifier, "init_speech_feature_extractor", side_effect=AssertionError("speech encoder")), patch.object(Classifier, "init_text_feature_extractor", side_effect=AssertionError("text encoder")):
    trainer = Trainer(params)
assert params.speech_feature_extractor_output_vectors_dimension == 4
assert params.text_feature_extractor_output_vectors_dimension == 4
text = torch.tensor([0.375])
assert trainer.text_input_to_device(text, "cpu").item() == 0.375
before = [p.detach().clone() for p in trainer.net.parameters()]
trainer.train(0, 1)
assert any(not torch.equal(a, b) for a, b in zip(before, trainer.net.parameters()))
trainer.evaluate_training()
trainer.evaluate_validation()
# Old parameter objects still retain the token-ID conversion path.
params.precomputed_features_dir = None
assert trainer.text_input_to_device(torch.tensor([2]), "cpu").dtype == torch.long
'''
        result = subprocess.run(
            [sys.executable, "-c", code,
             "--train_labels_path", str(self.csv), "--validation_labels_path", str(self.csv),
             "--precomputed_features_dir", str(self.root),
             "--log_file_folder", str(self.root / "logs"),
             "--model_output_folder", str(self.root / "models"),
             "--training_batch_size", "4", "--evaluation_batch_size", "3",
             "--classifier_hidden_layers_width", "8", "--classifier_hidden_layers", "0",
             "--eval_and_save_best_model_every", "0", "--print_training_info_every", "0"],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
