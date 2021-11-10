import logging
import os
from pathlib import Path
from typing import List

import pandas as pd
import torch
from datasets import Dataset

from argument_parsers import parser
from dataset_parsers import load_and_prepare_dataset
from hf.model import BertModel, BertTokenizerFast, MT5Model

logger = logging.getLogger(__name__)


class MT5DataProcessor:
    def __init__(self, tokenizer, max_source_length=512, max_target_length=80):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def process(self, dataset):
        dataset = dataset.map(self._convert_to_features, batched=True)

        return dataset

    # tokenize the examples
    def _convert_to_features(self, example_batch):
        source_encoding = self.tokenizer.batch_encode_plus(
            example_batch["source_text"],
            max_length=self.max_source_length,
            padding="max_length",
            pad_to_max_length=True,
            truncation=True,
        )
        target_encoding = self.tokenizer.batch_encode_plus(
            example_batch["target_text"],
            max_length=self.max_target_length,
            padding="max_length",
            pad_to_max_length=True,
            truncation=True,
        )

        encodings = {
            "source_ids": source_encoding["input_ids"],
            "target_ids": target_encoding["input_ids"],
            "attention_mask": source_encoding["attention_mask"],
        }

        return encodings


class BertDataProcessor:
    def __init__(self, tokenizer: BertTokenizerFast, max_source_length: int = 512):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length

    def process(self, dataset):
        dataset = dataset.map(self._convert_to_features, batched=True)

        return dataset

    def _add_token_positions(self, encodings, answers):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(encodings.char_to_token(i, answers[i]["answer_start"]))
            end_positions.append(encodings.char_to_token(i, answers[i]["answer_end"] - 1))

            # if start position is None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = self.max_source_length
            if end_positions[-1] is None:
                end_positions[-1] = self.max_source_length

        encodings.update({"start_positions": start_positions, "end_positions": end_positions})

    # tokenize the examples
    def _convert_to_features(self, example_batch):
        encodings = self.tokenizer(
            example_batch["context"],
            example_batch["question"],
            max_length=self.max_source_length,
            padding="max_length",
            pad_to_max_length=True,
            truncation=True,
        )

        self._add_token_positions(encodings, example_batch["answer"])

        return encodings


def _read_datasets(
    names: List[str], target_format="mt5", mt5_task_list: List[str] = ["ans_ext", "qa", "qg"], mt5_qg_format="highlight"
) -> Dataset:
    """
    Args:
        names: lisf of dataset subset names or paths
        target_format (str): output format ('mt5' or 'bert')
        mt5_task_list: list of tasks for mt5 data to be prepared
        mt5_qg_format: "highlight", "prepend" or "both"
    """
    data = []
    for name in names:
        data.extend(
            load_and_prepare_dataset(
                name, target_format=target_format, mt5_task_list=mt5_task_list, mt5_qg_format=mt5_qg_format
            )
        )
    data = Dataset.from_pandas(pd.DataFrame(data))
    return data


def main(args_file_path: str = None):
    model_args, data_args, train_args = parser(args_file_path)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
    )

    # set datasets
    train_dataset = _read_datasets(
        names=train_args.train_dataset_list,
        target_format=model_args.model_type,
        mt5_task_list=train_args.mt5_task_list,
        mt5_qg_format=train_args.mt5_qg_format,
    )
    valid_dataset = _read_datasets(
        names=train_args.valid_dataset_list,
        target_format=model_args.model_type,
        mt5_task_list=train_args.mt5_task_list,
        mt5_qg_format=train_args.mt5_qg_format,
    )

    # set tokenizer
    if model_args.model_type == "mt5":
        model = MT5Model(model_args.model_name_or_path)
        tokenizer = model.tokenizer
        tokenizer.add_tokens(["<sep>", "<hl>"])
    elif model_args.model_type == "bert":
        model = BertModel(model_args.model_name_or_path)
        tokenizer = model.tokenizer

    # set processor
    if model_args.model_type == "mt5":
        processor = MT5DataProcessor(
            tokenizer, max_source_length=train_args.max_source_length, max_target_length=train_args.max_target_length
        )
    elif model_args.model_type == "bert":
        processor = BertDataProcessor(tokenizer, max_source_length=train_args.max_source_length)

    # process datasets
    train_dataset = processor.process(train_dataset)
    valid_dataset = processor.process(valid_dataset)

    if model_args.model_type == "mt5":
        columns = ["source_ids", "target_ids", "attention_mask"]
        train_dataset.set_format(type="torch", columns=columns)
        valid_dataset.set_format(type="torch", columns=columns)
    elif model_args.model_type == "bert":
        columns = ["start_positions", "end_positions", "input_ids", "attention_mask"]
        train_dataset.set_format(type="torch")
        valid_dataset.set_format(type="torch")

    # create train/valid file dirs
    train_file_path = Path(str(data_args.train_file_path).strip())
    if not train_file_path.parent.exists():
        train_file_path.parent.mkdir(parents=True, exist_ok=True)
    valid_file_path = Path(str(data_args.valid_file_path).strip())
    if not valid_file_path.parent.exists():
        valid_file_path.parent.mkdir(parents=True, exist_ok=True)

    # save train/valid files
    torch.save(train_dataset, train_file_path)
    logger.info(f"saved train dataset at {train_file_path}")

    torch.save(valid_dataset, valid_file_path)
    logger.info(f"saved validation dataset at {valid_file_path}")

    # save tokenizer
    tokenizer_path = model_args.tokenizer_path
    if not os.path.exists(tokenizer_path):
        os.mkdir(tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"saved tokenizer at {tokenizer_path}")


if __name__ == "__main__":
    main()
