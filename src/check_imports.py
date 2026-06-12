"""Check that the project environment has the expected core packages."""

import accelerate
import datasets
import einops
import evaluate
import matplotlib
import numpy
import pandas
import scipy
import seaborn
import sklearn
import tokenizers
import torch
import tqdm
import transformers
import wandb
import yaml


def main() -> None:
    print("Core imports OK")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
