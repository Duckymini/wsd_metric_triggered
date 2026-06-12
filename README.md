# Automatic WSD Decay

This project studies whether the decay phase of warmup-stable-decay (WSD) learning-rate schedules for LLaMA-style language-model pretraining can be selected automatically from training dynamics, instead of relying only on a fixed decay start and fixed final learning-rate ratio.

The repository contains the training code, experiment configurations, launch scripts, logs, final metrics, and analysis notebooks used for the project. The saved logs and results are committed so the experiments can be inspected without rerunning every training job, but all runs can also be reproduced from the provided configs and scripts.

## Contents

- [Environment Setup](#environment-setup)
- [Running Experiments](#running-experiments)
- [Reproducibility Notes](#reproducibility-notes)
- [Repository Structure](#repository-structure)
- [Config, Log, and Result Formats](#config-log-and-result-formats)
- [Notebooks](#notebooks)
- [Source Files](#source-files)
- [Experiments Included](#experiments-included)
- [Inspecting Results](#inspecting-results)

---

## Environment Setup

The experiments are GPU-oriented. They were run with CUDA, and rerunning the full jobs on CPU or Apple MPS is not recommended because training will be very slow.

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate wsd-decay
```

There are two equivalent ways to launch an experiment. Choose one depending on whether you prefer running the Python command directly or using the provided bash scripts.

### Option A: Run Directly From the Terminal

This option does not require editing any `.sh` file. From the repository root, make the local `src/` package importable for the current terminal session, then run training:

```bash
cd /path/to/your/optimisation
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m src.train --config configs/wsd_intermediate_10k.yaml
```

The `export PYTHONPATH=...` line is not a repository file to edit. It is a terminal command that lets Python import modules from `src/`, such as `src.train`, `src.data`, and `src.model`.

If you want Hugging Face cache files to go to a specific directory, set the cache variables in the same terminal before launching training:

```bash
export HF_HOME=/path/with/enough/space/hf-cache
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
```

### Option B: Run With a Bash Script

The launch scripts in `scripts/*.sh` wrap the same Python command and set the environment variables for you. Before running a script on a new machine, adapt the machine-specific paths near the top of the script. For example, replace:

```bash
cd /scratch/optimisation
export PYTHONPATH=/scratch/optimisation:${PYTHONPATH:-}
export HF_HOME=/scratch/hf-cache
export TRANSFORMERS_CACHE=/scratch/hf-cache/transformers
export HF_DATASETS_CACHE=/scratch/hf-cache/datasets
```

with paths that match your machine:

```bash
cd /path/to/your/optimisation
export PYTHONPATH=/path/to/your/optimisation:${PYTHONPATH:-}
export HF_HOME=/path/with/enough/space/hf-cache
export TRANSFORMERS_CACHE=/path/with/enough/space/hf-cache/transformers
export HF_DATASETS_CACHE=/path/with/enough/space/hf-cache/datasets
```

Then run the script:

```bash
bash scripts/run_wsd_intermediate_10k.sh
```

For full reproducibility, keep the YAML config unchanged and only adapt machine-specific paths: the repository path in `cd`/`PYTHONPATH` and the Hugging Face cache path.

### Dataset and Cache Notes

The dataset is loaded through Hugging Face streaming. The current configs use `streaming: true` with `HuggingFaceFW/fineweb`, `config_name: sample-100BT`, and validation examples obtained from an offset in the training stream.

The cache variables are terminal environment variables, not config files. They tell Hugging Face where to store dataset metadata, tokenizer files, and cached artifacts. Use a directory that exists and has enough disk space. A local-machine example is:

```bash
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
```

If you want to use non-streaming datasets, local files, or a different train/validation split, update the `dataset` section in the YAML configs and, if needed, extend `src/data.py`.

---

## Running Experiments

After the environment is ready, launch a run either directly with `python -m src.train --config ...` or through one of the scripts in `scripts/`, following the setup option chosen above.

Each run creates a timestamped `run_id`:

```text
<run_name>_<YYYYMMDD-HHMMSS>
```

For example:

```text
wsd_intermediate_10k_20260606-071741
```

Training outputs are written to:

```text
logs/<run_id>/
results/<run_id>/
checkpoints/<run_id>/
```

The run is also appended to `experiments/registry.csv`.

---

## Reproducibility Notes

<table border="1" rules="all" frame="box" cellspacing="0" cellpadding="6">
  <thead>
    <tr>
      <th>Reproducibility item</th>
      <th>Where to find it</th>
      <th>Why it matters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Experiment definitions</td>
      <td><code>configs/*.yaml</code></td>
      <td>Define the dataset, model size, training hyperparameters, schedule, seed, and policy settings.</td>
    </tr>
    <tr>
      <td>Saved run config</td>
      <td><code>results/&lt;run_id&gt;/config.yaml</code> and <code>results/&lt;run_id&gt;/source_config.yaml</code></td>
      <td>Keep a copy of the config used to produce each completed run.</td>
    </tr>
    <tr>
      <td>Final metrics</td>
      <td><code>results/&lt;run_id&gt;/final_metrics.json</code></td>
      <td>Stores final validation loss, token count, final learning rate, runtime, device, and schedule type.</td>
    </tr>
    <tr>
      <td>Training logs</td>
      <td><code>logs/&lt;run_id&gt;/metrics.jsonl</code></td>
      <td>Stores step-level training metrics and periodic validation metrics.</td>
    </tr>
    <tr>
      <td>Run registry</td>
      <td><code>experiments/registry.csv</code></td>
      <td>Gives a compact overview of completed runs and final validation losses.</td>
    </tr>
    <tr>
      <td>Saved logs/results</td>
      <td><code>logs/</code> and <code>results/</code></td>
      <td>Committed so the experiments can be inspected without rerunning expensive jobs.</td>
    </tr>
    <tr>
      <td>Rerunning experiments</td>
      <td><code>scripts/*.sh</code> or <code>python -m src.train --config ...</code></td>
      <td>Runs can be regenerated from the provided configs after adapting only machine-specific paths.</td>
    </tr>
    <tr>
      <td>Checkpoints</td>
      <td><code>checkpoints/</code></td>
      <td>The folder is kept, but <code>.pt</code> files are ignored because checkpoints are large. Reruns can still write checkpoints there.</td>
    </tr>
    <tr>
      <td>Random seed</td>
      <td><code>seed</code> field in each config</td>
      <td>The code calls <code>transformers.set_seed</code>; small differences may still occur across hardware, CUDA/PyTorch versions, and streaming order.</td>
    </tr>
  </tbody>
</table>

---

## Repository Structure

```text
optimisation/
├── configs/        # experiment YAML files
├── scripts/        # launch scripts
├── src/            # training source code
├── logs/           # committed run logs
├── results/        # committed final run outputs
├── experiments/    # run registry
├── notebooks/      # analysis notebooks
├── checkpoints/    # local checkpoints, ignored by Git
└── environment.yml # Conda environment
```

<table border="1" rules="all" frame="box" cellspacing="0" cellpadding="6">
  <thead>
    <tr>
      <th>Path</th>
      <th>Contents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>configs/</code></td>
      <td>YAML experiment configs. Each file defines the run name, seed, dataset, model size, training hyperparameters, and schedule/policy settings.</td>
    </tr>
    <tr>
      <td><code>scripts/</code></td>
      <td>Bash launchers for the main experiments. They set environment variables and call <code>python -m src.train --config ...</code>.</td>
    </tr>
    <tr>
      <td><code>src/</code></td>
      <td>Python source code; individual files are described in the Source Files section.</td>
    </tr>
    <tr>
      <td><code>logs/</code></td>
      <td>Committed run logs used for analysis and reproducibility.</td>
    </tr>
    <tr>
      <td><code>results/</code></td>
      <td>Committed final run artifacts, including saved configs and final metrics.</td>
    </tr>
    <tr>
      <td><code>experiments/</code></td>
      <td><code>registry.csv</code>, a compact table of completed runs and final validation losses.</td>
    </tr>
    <tr>
      <td><code>notebooks/</code></td>
      <td>Analysis notebooks used to visualise the saved logs and results.</td>
    </tr>
    <tr>
      <td><code>checkpoints/</code></td>
      <td>Empty in Git except <code>.gitkeep</code>; used for local <code>.pt</code> checkpoints during reruns.</td>
    </tr>
    <tr>
      <td><code>environment.yml</code></td>
      <td>Conda environment specification.</td>
    </tr>
  </tbody>
</table>

---

## Config, Log, and Result Formats

### Config File Format

Configs are YAML files with this general structure:

```yaml
run_name: wsd_intermediate_10k
owner: TBD
seed: 1

dataset:
  name: HuggingFaceFW/fineweb
  config_name: sample-100BT
  tokenizer_name: gpt2
  streaming: true

model:
  name: llama_intermediate_512x8
  context_length: 1024
  hidden_size: 512
  num_hidden_layers: 8

training:
  max_steps: 10000
  batch_size: 8
  gradient_accumulation_steps: 64
  learning_rate: 6.0e-4

schedule:
  type: wsd_s
  warmup_steps: 100
  final_lr_ratio: 0.1
```

Policy-decay configs add a `training.policy_decay` block with the policy name, trigger constraints, thresholds, final LR ratio, and decay type.

### Log File Format

The main log is newline-delimited JSON:

```text
logs/<run_id>/metrics.jsonl
```

Typical training rows contain:

```json
{"step": 5, "tokens_seen": 2621440, "train_loss": 10.7092, "loss_variance": 0.0053, "grad_norm": 2.3457, "grad_snr": 0.7245, "learning_rate": 0.000036, "elapsed_seconds": 30.9}
```

Periodic validation rows contain:

```json
{"step": 50, "tokens_seen": 26214400, "validation_loss": 7.1234, "learning_rate": 0.000306, "elapsed_seconds": 290.1}
```

Some runs also include:

<table border="1" rules="all" frame="box" cellspacing="0" cellpadding="6">
  <thead>
    <tr>
      <th>File</th>
      <th>Meaning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>policy_trigger.json</code></td>
      <td>Policy name, trigger step, decay length, final LR ratio, thresholds, and metrics at trigger time.</td>
    </tr>
    <tr>
      <td><code>decay_amount_sweep.jsonl</code></td>
      <td>Probe summary for candidate final LR ratios at selected plateau steps.</td>
    </tr>
    <tr>
      <td><code>decay_amount_trajectory.jsonl</code></td>
      <td>Intermediate validation trajectory during each temporary probe decay.</td>
    </tr>
  </tbody>
</table>

### Result File Format

Each result directory has:

<table border="1" rules="all" frame="box" cellspacing="0" cellpadding="6">
  <thead>
    <tr>
      <th>File</th>
      <th>Meaning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>config.yaml</code></td>
      <td>Config saved at runtime.</td>
    </tr>
    <tr>
      <td><code>source_config.yaml</code></td>
      <td>Copy of the source YAML passed to <code>--config</code>.</td>
    </tr>
    <tr>
      <td><code>final_metrics.json</code></td>
      <td>Final scalar metrics for the run.</td>
    </tr>
  </tbody>
</table>

Example `final_metrics.json`:

```json
{
  "run_id": "wsd_intermediate_10k_20260606-071741",
  "final_train_loss": 3.8540,
  "final_validation_loss": 3.4422,
  "tokens_seen": 5242880000,
  "final_learning_rate": 0.000006,
  "total_seconds": 55987.5,
  "device": "cuda",
  "schedule": "wsd_s"
}
```

---

## Notebooks

The notebooks allow you to visualise the saved logs and result files without rerunning the training jobs.

<table border="1" rules="all" frame="box" cellspacing="0" cellpadding="6">
  <thead>
    <tr>
      <th>Notebook</th>
      <th>Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>notebooks/wsd_vs_cosine.ipynb</code></td>
      <td>Compares the saved WSD and cosine runs using their validation losses and learning-rate trajectories.</td>
    </tr>
    <tr>
      <td><code>notebooks/policy_decay_comparison.ipynb</code></td>
      <td>Analyses the policy-decay runs against the fixed WSD baseline.</td>
    </tr>
    <tr>
      <td><code>notebooks/plateau_decay_amount_metrics_analysis.ipynb</code></td>
      <td>Inspects the plateau-region decay-amount sweep and the metrics recorded during probe decays.</td>
    </tr>
  </tbody>
</table>

---

## Source Files

<table border="1" rules="all" frame="box" cellspacing="0" cellpadding="6">
  <thead>
    <tr>
      <th>File</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>src/__init__.py</code></td>
      <td>Marks <code>src</code> as a Python package so modules can be run with <code>python -m src.train</code>.</td>
    </tr>
    <tr>
      <td><code>src/config.py</code></td>
      <td>Loads YAML experiment configs and saves resolved configs to result directories.</td>
    </tr>
    <tr>
      <td><code>src/data.py</code></td>
      <td>Loads Hugging Face language-model datasets, supports streaming mode, tokenizes text, and packs tokens into fixed-length causal-LM blocks.</td>
    </tr>
    <tr>
      <td><code>src/model.py</code></td>
      <td>Builds a randomly initialized LLaMA causal language model from config hyperparameters and tokenizer metadata.</td>
    </tr>
    <tr>
      <td><code>src/policy.py</code></td>
      <td>Implements metric-based decay trigger policies: low loss variance, low gradient SNR, and their conjunction.</td>
    </tr>
    <tr>
      <td><code>src/schedules.py</code></td>
      <td>Builds learning-rate schedules for warmup-stable, WSD, WSD-beta, cosine decay, policy-triggered decay, and temporary probe decays.</td>
    </tr>
    <tr>
      <td><code>src/train.py</code></td>
      <td>Main training entrypoint. It prepares outputs, loads data/model, trains, evaluates, logs metrics, saves checkpoints, runs policy triggers or decay-amount probes, writes final metrics, and updates the run registry.</td>
    </tr>
  </tbody>
</table>

---

## Experiments Included

<table border="1" rules="all" frame="box" cellspacing="0" cellpadding="6">
  <thead>
    <tr>
      <th>Config</th>
      <th>Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>configs/wsd_intermediate_10k.yaml</code></td>
      <td>Fixed WSD baseline with decay near the end of a 10k-step run.</td>
    </tr>
    <tr>
      <td><code>configs/cosine_intermediate_10k.yaml</code></td>
      <td>Cosine learning-rate baseline for comparison.</td>
    </tr>
    <tr>
      <td><code>configs/wsd_beta_intermediate_10k.yaml</code></td>
      <td>WSD run with learning-rate decay and Adam beta co-decay.</td>
    </tr>
    <tr>
      <td><code>configs/wsd_intermediate_plateau_amount_sweep_5k.yaml</code></td>
      <td>Plateau-region probes comparing several final LR ratios.</td>
    </tr>
    <tr>
      <td><code>configs/policy_decay_baseline_7k.yaml</code></td>
      <td>Fixed 7k-step WSD baseline for policy-trigger experiments.</td>
    </tr>
    <tr>
      <td><code>configs/policy_decay_loss_variance_7k.yaml</code></td>
      <td>Automatic decay triggered by low loss variance.</td>
    </tr>
    <tr>
      <td><code>configs/policy_decay_grad_snr_7k.yaml</code></td>
      <td>Automatic decay triggered by low gradient SNR.</td>
    </tr>
    <tr>
      <td><code>configs/policy_decay_combined_7k.yaml</code></td>
      <td>Automatic decay triggered when both loss variance and gradient SNR conditions hold.</td>
    </tr>
  </tbody>
</table>

---

## Inspecting Results

The fastest terminal overview is:

```bash
cat experiments/registry.csv
```

For plots and detailed comparisons, use the notebooks listed above. They consume the committed logs and results, so they can be run without retraining as long as the environment has the dependencies from `environment.yml`.
