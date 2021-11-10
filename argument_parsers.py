import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from transformers import HfArgumentParser, TrainingArguments

from utils.file import read_yaml

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="google/mt5-small",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    tokenizer_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer path to be saved/loaded"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    wandb_project: Optional[str] = field(default=None, metadata={"help": "Wandb project for experiment tracking"})
    wandb_id: Optional[str] = field(
        default=None, metadata={"help": "Wandb run id, will be used when 'do_train=False', 'do_eval=True'"}
    )
    neptune_project: Optional[str] = field(default=None, metadata={"help": "Neptune project for experiment tracking"})
    neptune_run: Optional[str] = field(
        default=None, metadata={"help": "Neptune run id, will be used when 'do_train=False', 'do_eval=True'"}
    )
    neptune_api_token: Optional[str] = field(
        default=None, metadata={"help": "Neptune api token for experiment tracking"}
    )
    model_type: str = field(default=None, metadata={"help": "'mt5' or 'bert'"})


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file_path: Optional[str] = field(
        default="data/train_data_multitask_mt5.pt",
        metadata={"help": "name for cached train dataset"},
    )
    valid_file_path: Optional[str] = field(
        default="data/valid_data_multitask_mt5.pt",
        metadata={"help": "name for cached valid dataset"},
    )

    prepare_data: bool = field(
        default=True,
        metadata={
            "help": "Runs prepare_data.py before starting to train. Set if false if you alreade prepared data before."
        },
    )


@dataclass
class ExtendedTrainingArguments(TrainingArguments):
    train_dataset_list: Optional[List[str]] = field(
        default_factory=lambda: ["tquad-train"],
        metadata={"help": "dataset name list of the training"},
    )
    valid_dataset_list: Optional[List[str]] = field(
        default_factory=lambda: ["tquad-val"],
        metadata={"help": "dataset name list of the validation"},
    )
    eval_dataset_list: Optional[List[str]] = field(
        default_factory=lambda: ["tquad-val", "xquad.tr"],
        metadata={"help": "dataset name list of the evaluation"},
    )
    freeze_embeddings: bool = field(
        default=False,
        metadata={"help": "Freeze token embeddings."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    max_target_length: Optional[int] = field(
        default=80,
        metadata={"help": "Max input length for the target text"},
    )
    mt5_task_list: Optional[List[str]] = field(
        default_factory=lambda: ["qa", "qg", "ans_ext"],
        metadata={"help": "task list for mt5"},
    )
    mt5_qg_format: Optional[str] = field(
        default="highlight",
        metadata={"help": 'mt5 qg format as "highlight", "prepend" or "both"'},
    )


def parser(args_file_path: Optional[str] = None):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataArguments, ExtendedTrainingArguments))

    accepted_args_file_suffixes = (".json", ".yml", ".yaml")

    # validate args file suffix
    args_file_suffix: Optional[str] = None
    if args_file_path:
        # If we pass args_file_path to the script and it's the path to a json/yml/yaml file,
        # let's parse it to get our arguments.
        assert type(args_file_path) == str, TypeError(f"invalid 'args_file_path': {args_file_path}")
        args_file_suffix = Path(args_file_path).suffix
        assert args_file_suffix in accepted_args_file_suffixes, TypeError(
            f"""args file should be one of: /
            {accepted_args_file_suffixes}, invalid args file format: {args_file_suffix}"""
        )
    elif len(sys.argv) == 2:
        # If we pass only one argument to the script and it's the path to a json/yml/yaml file,
        # let's parse it to get our arguments.
        args_file_path = sys.argv[1]
        args_file_suffix = Path(args_file_path).suffix
        assert args_file_suffix in accepted_args_file_suffixes, TypeError(
            f"""args file should be one of: /
            {accepted_args_file_suffixes}, invalid args file format: {args_file_suffix}"""
        )

    if args_file_suffix == ".json":
        model_args, data_args, training_args = parser.parse_json_file(json_file=args_file_path)
    elif args_file_suffix in (".yml", ".yaml"):
        args_dict = read_yaml(args_file_path)
        model_args, data_args, training_args = parser.parse_dict(args=args_dict)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.tokenizer_path is None:
        model_name = (model_args.model_name_or_path).split("/")[-1]
        model_args.tokenizer_path = model_name + "_tokenizer"

    # overwrite model type
    if "mt5" in model_args.model_name_or_path:
        model_type = "mt5"
    elif "bert" in model_args.model_name_or_path:
        model_type = "bert"
    else:
        logger.info("couldnt infer model type from 'model_name_or_path', assuming its 'mt5'.")
        model_type = "mt5"
    model_args.model_type = model_type

    return model_args, data_args, training_args
