import json
import logging
import os
from typing import Tuple

import torch
import transformers
from transformers import Trainer as HFTrainer
from transformers import set_seed
from transformers.hf_argparser import DataClass
from transformers.optimization import Adafactor, AdamW
from transformers.trainer import Trainer

from core.argument_parsers import parser
from core.collator import T2TDataCollator
from core.evaluate import evaluate_on_train_end
from hf.model import BertModel, MT5Model
from prepare_data import main as prepare_data
from utils.file import save_experiment_config
from utils.neptune import init_neptune, log_to_neptune
from utils.wandb import init_wandb, log_to_wandb


def setup_logger(args: DataClass) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper() if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    logger.info("Training/evaluation parameters %s", args)
    return logger


def check_output(args: DataClass, logger: logging.Logger = None) -> None:
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )


def load_datasets(args: DataClass, train: bool, eval: bool, logger: logging.Logger) -> Tuple:
    logger.info("loading dataset")
    train_dataset = torch.load(args.train_file_path) if train else None
    valid_dataset = torch.load(args.valid_file_path) if eval else None
    logger.info("finished loading dataset")

    return train_dataset, valid_dataset


def main(args_file_path: str = None):

    model_args, data_args, training_args = parser(args_file_path)

    # check for output_dir with given arguments.
    check_output(training_args)

    logger = setup_logger(training_args)

    # set seed
    set_seed(training_args.seed)

    # initialize experiment tracking
    report_to = []

    if training_args.do_train:
        wandb_status, wandb = init_wandb(project=model_args.wandb_project, name=training_args.run_name)
    else:
        wandb_status, wandb = init_wandb(
            project=model_args.wandb_project, name=training_args.run_name, id=model_args.wandb_id
        )
    neptune_status, neptune = init_neptune(
        project=model_args.neptune_project, api_token=model_args.neptune_api_token, name=training_args.run_name
    )

    if wandb_status:
        report_to.append("wandb")
    if neptune_status:
        report_to.append("neptune")

    training_args.report_to = report_to

    # disable wandb console logs
    logging.getLogger("wandb.run_manager").setLevel(logging.WARNING)

    # prepare data()
    if data_args.prepare_data:
        prepare_data(args_file_path)

    # load model
    if model_args.model_type == "mt5":
        model = MT5Model(
            model_name_or_path=model_args.model_name_or_path,
            tokenizer_name_or_path=model_args.tokenizer_path,
            freeze_embeddings=training_args.freeze_embeddings,
            cache_dir=model_args.cache_dir,
            use_cuda=True,
        )
    elif model_args.model_type == "bert":
        model = BertModel(
            model_name_or_path=model_args.model_name_or_path,
            tokenizer_name_or_path=model_args.tokenizer_path,
            freeze_embeddings=training_args.freeze_embeddings,
            cache_dir=model_args.cache_dir,
            use_cuda=True,
        )
    train_dataset, valid_dataset = load_datasets(
        data_args, train=training_args.do_train, eval=training_args.do_eval, logger=logger
    )

    # set optimizer
    if training_args.adafactor:
        # as adviced in https://huggingface.co/transformers/main_classes/optimizer_schedules.html#adafactor-pytorch
        optimizer = Adafactor(
            model.model.parameters(),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=training_args.weight_decay,
            lr=training_args.learning_rate,
        )
    else:
        optimizer = AdamW(
            model.model.parameters(), weight_decay=training_args.weight_decay, lr=training_args.learning_rate
        )

    if model_args.model_type == "mt5":
        # initialize data_collator
        data_collator = T2TDataCollator(
            tokenizer=model.tokenizer, mode="training", using_tpu=training_args.tpu_num_cores is not None
        )

    # fix https://discuss.huggingface.co/t/mt5-fine-tuning-keyerror-source-ids/5257/2
    training_args.remove_unused_columns = False if model_args.model_type == "mt5" else True

    # export experiment config
    save_experiment_config(model_args, data_args, training_args)

    # start training
    if training_args.do_train:
        # init model
        if model_args.model_type == "mt5":
            trainer: Trainer = HFTrainer(
                model=model.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                data_collator=data_collator,
                optimizers=(optimizer, None),
            )
        elif model_args.model_type == "bert":
            trainer: Trainer = HFTrainer(
                model=model.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                optimizers=(optimizer, None),
            )

        # perform training
        trainer.train(
            resume_from_checkpoint=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)

        model.tokenizer.save_pretrained(training_args.output_dir)

    # start evaluation
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        # arange neptune/wandb loggers
        if training_args.do_train:
            for callback in trainer.callback_handler.callbacks:
                if isinstance(callback, transformers.integrations.WandbCallback):
                    wandb = callback._wandb
            for callback in trainer.callback_handler.callbacks:
                if isinstance(callback, transformers.integrations.NeptuneCallback):
                    neptune_run = callback._neptune_run
        if not training_args.do_train:
            if "neptune" in report_to:
                neptune_run = neptune.init(
                    project=os.getenv("NEPTUNE_PROJECT"),
                    api_token=os.getenv("NEPTUNE_API_TOKEN"),
                    mode=os.getenv("NEPTUNE_CONNECTION_MODE", "async"),
                    name=os.getenv("NEPTUNE_RUN_NAME", None),
                    run=model_args.neptune_run,
                )
            elif "wandb" in report_to:
                wandb.init(project=model_args.wandb_project, name=model_args.run_name, id=model_args.wandb_id)

        # calculate evaluation results
        overall_results = evaluate_on_train_end(model_args, training_args)

        # log to neptune/wandb
        if "neptune" in report_to:
            log_to_neptune(neptune_run, overall_results)
        if "wandb" in report_to:
            log_to_wandb(wandb, overall_results)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


def run_multi(args_dict):
    with open("args.json", "w") as f:
        json.dump(args_dict, f)

    main(args_file="args.json")


if __name__ == "__main__":
    main()
