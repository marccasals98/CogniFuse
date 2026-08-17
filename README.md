# CogniFuse

Predict Cognitive decliness.

## How to use
CogniFuse uses uv to manage Python versions, dependencies, virtual environments, and reproducible installations.

Install uv and run the following command before executing:

```bash
uv sync --locked
```

## Repository organization
The main folders of the repo are the following:

* `/scripts`: The main scripts following classical PyTorch file structure.
* `/shs`: Scripts designed to launch experiments in HPC systems (using SLURM).
* `/utils`: Auxiliary stand-alone files that have no direct impact on the training.