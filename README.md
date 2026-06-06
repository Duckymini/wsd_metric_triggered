# Automatic WSD Decay

Project question: can WSD learning-rate decay be chosen automatically from training metrics instead of using a fixed decay time and fixed decay amount?

## Structure

```text
.
├── environment.yml       # local development environment
├── environment-h200.yml  # GPU training environment
├── configs/          # experiment config files
├── src/              # training/evaluation code later
├── scripts/          # small runnable utilities later
├── experiments/      # run registry and experiment notes
├── results/          # final tables and figures
├── notebooks/        # analysis notebooks
├── reports/          # report and slides
├── data/             # local datasets, not committed
├── logs/             # local logs, not committed
└── checkpoints/      # local checkpoints, not committed
```

## Experiment Plan

1. Debug a small 30M model pipeline.
2. Compare cosine LR with fixed WSD.
3. Test final LR ratios: `0.3`, `0.1`, `0.03`, `0.01`.
4. Test metric-based decay triggers.
5. Analyze whether pre-decay metrics predict the best decay amount.
6. Optionally confirm one setting on a 0.1B model.

## Fixed WSD Baseline

- Warmup, stable high LR, decay.
- Decay starts at 90% of total steps.
- Decay lasts 10% of total steps.
- Final LR is 0.1 times peak LR.

## Team Workflow

- Add every run to `experiments/registry.csv`.
- Keep large files out of Git: datasets, logs, checkpoints.
- Commit configs and final result summaries.
- Use clear run names, for example:

```text
p03_30m_fineweb_wsd090_ratio010_seed001
```

## Environment

Recommended setup on your Mac:

```bash
conda env create -f environment.yml
conda activate wsd-decay
python -m src.check_imports
```

Setup on the H200 GPU machine:

```bash
conda env create -f environment-h200.yml
conda activate wsd-decay-h200
python -m src.check_imports
```

Use `environment.yml` locally and `environment-h200.yml` only on the Linux GPU machine.

## Baseline Runs

Launch a WSD debug run:

```bash
bash scripts/run_wsd_debug.sh
```

Launch a cosine debug run:

```bash
bash scripts/run_cosine_debug.sh
```

Equivalent direct commands:

```bash
python -m src.train --config configs/wsd_debug.yaml
python -m src.train --config configs/cosine_debug.yaml
```

Each run writes:

```text
checkpoints/<run_id>/checkpoint_step_*.pt
logs/<run_id>/metrics.jsonl
results/<run_id>/config.yaml
results/<run_id>/source_config.yaml
results/<run_id>/final_metrics.json
```

The run is also appended to `experiments/registry.csv`.
