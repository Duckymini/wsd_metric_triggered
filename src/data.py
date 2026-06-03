from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase


def _pick_text_column(dataset: Dataset, configured_column: str | None) -> str:
    if configured_column is not None:
        if configured_column not in dataset.column_names:
            raise ValueError(f"Text column '{configured_column}' not found in {dataset.column_names}.")
        return configured_column

    for column in dataset.column_names:
        first_value = dataset[0][column]
        if isinstance(first_value, str):
            return column
    raise ValueError(f"Could not find a text column in {dataset.column_names}.")


def _select_first(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return dataset
    return dataset.select(range(min(max_samples, len(dataset))))


def load_lm_datasets(
    dataset_config: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    context_length: int,
    seed: int,
) -> DatasetDict:
    dataset_name = dataset_config.get("name", "JeanKaddour/minipile")
    validation_split = float(dataset_config.get("validation_split", 0.02))
    raw = load_dataset(dataset_name)

    if "train" not in raw:
        first_split = next(iter(raw.keys()))
        raw = DatasetDict({"train": raw[first_split]})

    if "validation" not in raw:
        split = raw["train"].train_test_split(test_size=validation_split, seed=seed)
        raw = DatasetDict({"train": split["train"], "validation": split["test"]})

    train_raw = _select_first(raw["train"], dataset_config.get("max_train_samples"))
    valid_raw = _select_first(raw["validation"], dataset_config.get("max_validation_samples"))
    text_column = _pick_text_column(train_raw, dataset_config.get("text_column"))

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch[text_column], return_attention_mask=False)

    def group_texts(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated = []
        for ids in batch["input_ids"]:
            concatenated.extend(ids)
        total_length = (len(concatenated) // context_length) * context_length
        chunks = [
            concatenated[i : i + context_length]
            for i in range(0, total_length, context_length)
        ]
        return {"input_ids": chunks, "labels": [chunk.copy() for chunk in chunks]}

    remove_train_columns = train_raw.column_names
    remove_valid_columns = valid_raw.column_names
    tokenized_train = train_raw.map(
        tokenize,
        batched=True,
        remove_columns=remove_train_columns,
        desc="Tokenizing train split",
    )
    tokenized_valid = valid_raw.map(
        tokenize,
        batched=True,
        remove_columns=remove_valid_columns,
        desc="Tokenizing validation split",
    )

    lm_train = tokenized_train.map(group_texts, batched=True, desc="Grouping train tokens")
    lm_valid = tokenized_valid.map(group_texts, batched=True, desc="Grouping validation tokens")
    lm_train.set_format(type="torch")
    lm_valid.set_format(type="torch")
    return DatasetDict({"train": lm_train, "validation": lm_valid})

