from __future__ import annotations

from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from torch.utils.data import get_worker_info
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


class StreamingTokenBlockDataset(TorchIterableDataset):
    def __init__(
        self,
        source: Any,
        tokenizer: PreTrainedTokenizerBase,
        text_column: str,
        context_length: int,
        max_samples: int | None = None,
    ) -> None:
        self.source = source
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.context_length = context_length
        self.max_samples = max_samples

    def __iter__(self):
        worker = get_worker_info()
        buffer: list[int] = []
        samples_seen = 0

        for sample_idx, sample in enumerate(self.source):
            if worker is not None and sample_idx % worker.num_workers != worker.id:
                continue
            if self.max_samples is not None and samples_seen >= self.max_samples:
                break

            text = sample.get(self.text_column)
            if not isinstance(text, str):
                continue
            samples_seen += 1

            tokenized = self.tokenizer(text, return_attention_mask=False)
            buffer.extend(tokenized["input_ids"])
            while len(buffer) >= self.context_length:
                chunk = buffer[: self.context_length]
                buffer = buffer[self.context_length :]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": input_ids, "labels": input_ids.clone()}


def _expand_data_files(spec: Any) -> Any:
    if isinstance(spec, dict) and "url_template" in spec:
        start = int(spec.get("start", 0))
        end = int(spec["end"])
        return [spec["url_template"].format(i=i) for i in range(start, end)]
    return spec


def _resolve_data_files(data_files_config: dict[str, Any] | None) -> dict[str, Any] | None:
    if data_files_config is None:
        return None
    return {split: _expand_data_files(spec) for split, spec in data_files_config.items()}


def _load_streaming_split(
    loader_name: str,
    dataset_config_name: str | None,
    split_name: str,
    seed: int,
    shuffle_buffer_size: int | None,
    data_files: dict[str, Any] | None,
):
    if data_files is None:
        dataset = load_dataset(loader_name, dataset_config_name, split=split_name, streaming=True)
    else:
        dataset = load_dataset(loader_name, data_files=data_files, split=split_name, streaming=True)
    if shuffle_buffer_size is not None and shuffle_buffer_size > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
    return dataset


def load_streaming_lm_datasets(
    dataset_config: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    context_length: int,
    seed: int,
) -> dict[str, StreamingTokenBlockDataset]:
    dataset_name = dataset_config.get("name", "HuggingFaceFW/fineweb")
    loader_name = dataset_config.get("loader", dataset_name)
    dataset_config_name = dataset_config.get("config_name")
    data_files = _resolve_data_files(dataset_config.get("data_files"))
    text_column = dataset_config.get("text_column", "text")
    train_split = dataset_config.get("train_split", "train")
    validation_split = dataset_config.get("validation_split_name", "validation")
    shuffle_buffer_size = dataset_config.get("shuffle_buffer_size", 10_000)

    train_raw = _load_streaming_split(
        loader_name,
        dataset_config_name,
        train_split,
        seed,
        shuffle_buffer_size,
        data_files,
    )
    if dataset_config.get("validation_from_train", False):
        valid_offset = int(dataset_config.get("validation_stream_offset", 100_000))
        if data_files is None:
            valid_raw = load_dataset(
                loader_name,
                dataset_config_name,
                split=train_split,
                streaming=True,
            ).skip(valid_offset)
        else:
            valid_raw = load_dataset(
                loader_name,
                data_files=data_files,
                split=train_split,
                streaming=True,
            ).skip(valid_offset)
    else:
        try:
            valid_raw = _load_streaming_split(
                loader_name,
                dataset_config_name,
                validation_split,
                seed,
                None,
                data_files,
            )
        except Exception:
            valid_offset = int(dataset_config.get("validation_stream_offset", 100_000))
            if data_files is None:
                valid_raw = load_dataset(
                    loader_name,
                    dataset_config_name,
                    split=train_split,
                    streaming=True,
                ).skip(valid_offset)
            else:
                valid_raw = load_dataset(
                    loader_name,
                    data_files=data_files,
                    split=train_split,
                    streaming=True,
                ).skip(valid_offset)

    return {
        "train": StreamingTokenBlockDataset(
            train_raw,
            tokenizer,
            text_column,
            context_length,
            dataset_config.get("max_train_samples"),
        ),
        "validation": StreamingTokenBlockDataset(
            valid_raw,
            tokenizer,
            text_column,
            context_length,
            dataset_config.get("max_validation_samples"),
        ),
    }

def load_lm_datasets(
    dataset_config: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    context_length: int,
    seed: int,
) -> DatasetDict | dict[str, StreamingTokenBlockDataset]:
    if dataset_config.get("streaming", False):
        return load_streaming_lm_datasets(dataset_config, tokenizer, context_length, seed)

    dataset_name = dataset_config.get("name", "HuggingFaceFW/fineweb")
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
