# Checkpoint evaluation

`scripts/evaluate.py` evaluates an explicitly selected checkpoint from
`scripts/train.py` on prepared audio/text features. It reuses the classifier,
feature/mask validation, and batching without changing training code. It
supports the existing precomputed workflow, not raw-audio checkpoints.

```bash
uv run python scripts/evaluate.py \
  --checkpoint /absolute/path/to/model.chkpt \
  --dataset_to_evaluate outputs/adresso_full_whisper_distil_pauses/test/embeddings \
  --labels_path outputs/adresso_evaluation/official_labels/test_results_task1_groundtruth.csv \
  --bootstrap_samples 10000 --bootstrap_seed 1234 \
  --output_dir outputs/adresso_evaluation/new_run
```

Run from the repository root. `--checkpoint` is mandatory: no newest/best
checkpoint is selected by scanning model directories. The saved architecture
and weights are reconstructed exactly, with strict state-dictionary loading,
evaluation mode, frozen parameters, and inference mode. No optimizer runs.

For a separate GPU job, pass the same arguments to the new launcher:

```bash
sbatch shs/calcula/evaluate.sh \
  --checkpoint /absolute/path/to/model.chkpt \
  --dataset_to_evaluate outputs/adresso_full_whisper_distil_pauses/test/embeddings \
  --labels_path outputs/adresso_evaluation/official_labels/test_results_task1_groundtruth.csv \
  --output_dir outputs/adresso_evaluation/new_gpu_run
```

Slurm writes `evaluate_<jobid>.out` in the repository root. The launcher does
not scan for or choose a checkpoint, and the training launcher is unchanged.

`--dataset_to_evaluate` is the **prepared feature directory**. Every paired
recording under it is evaluated, including class subdirectories when present.
The checkpoint's audio/text suffixes and vector dimensions are respected;
missing pairs or masks, duplicate IDs, and incompatible dimensions fail.
The evaluator does not apply training folds or load a training dataset.

Labels are a separate CSV. Default recording-ID columns include `uid`, `ID`,
and `filename`; label columns include `label`, `diagnosis`, `Dx`, and
`DX_Pilar`. Use `--id_column`/`--label_column` for other schemas. Numeric class
IDs or known class names are accepted. ADReSSo's official `Control` and
`ProbableAD` map to `cn=0` and `ad=1`. Labels must cover exactly the evaluated
IDs and contain no duplicates. Rows are matched by ID, not CSV order.

Without `--labels_path`, only inference runs. No F-score or bootstrap values
are fabricated for unlabeled data. The local blank ADReSSo prediction template
is not ground truth. The official test-label archive is linked from the
[challenge page](https://luzs.gitlab.io/adresso-2021/). A separate verified copy
is under `outputs/adresso_evaluation/official_labels/`, with its source URL.

The script writes a new output directory and refuses existing ones:

- `predictions.csv`: recording IDs, predicted classes, probabilities, and
  ground truth when supplied.
- `metrics.json`: original-sample macro-F1, accuracy, class F1, confusion
  matrix, and bootstrap confidence bounds when labels are available.
- `manifest.json`: exact checkpoint path/hash, saved configuration/epoch,
  dataset IDs, input hashes, label-file hash, class mapping, and run settings.
- `bootstrap_scores.csv`: every replicate's macro-F1 when scoring.

Bootstrap uses paired `(true label, prediction)` resampling with replacement.
The default is 10,000 recording resamples, seed 1234, and a 95% percentile
confidence interval. Every resample uses the same complete class list and
`zero_division=0`. The original-sample macro-F1 is the primary score; the
bootstrap mean is not substituted for it. Inference runs once, regardless of
the number of resamples. This interval describes evaluation-sample uncertainty
conditional on this checkpoint, not variability between training runs.

For repeated recordings from a participant, supply a participant-ID column
using `--group_column patient_id`. Whole groups are then sampled together.
ADReSSo evaluation defaults to recording resampling; independent participant
identity is not established by the recording filename alone.

Training and evaluation remain separate. After a training run, pass its exact
checkpoint to this script. For CV, evaluate each explicitly selected fold
checkpoint into a different output directory, and do not select a model by
its test score. A future automatic training-launcher call should receive a
checkpoint path from that training run, not scan for a latest checkpoint.

```bash
TMPDIR=/tmp uv run python -m unittest discover -s evaluation_tests -v
TMPDIR=/tmp uv run python -m unittest discover -s tests -v
```
