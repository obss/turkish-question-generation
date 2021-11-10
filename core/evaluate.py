import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from jury import Jury
from jury.metrics import load_metric

from core.dataset_parsers import load_dataset
from core.generate import generate
from utils.file import load_json, save_json

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)


TASK_TO_METRIC = {
    "ans_ext": [
        load_metric("f1", task="language-generation"),
        load_metric("precision", task="language-generation"),
        load_metric("bertscore", compute_kwargs={"lang": "tr"}),
    ],
    "qa": [load_metric("squad"), load_metric("bertscore", compute_kwargs={"lang": "tr"})],
    "qg": [
        load_metric("bleu", compute_kwargs={"max_order": 1}),
        load_metric("bleu", compute_kwargs={"max_order": 2}),
        load_metric("bleu", compute_kwargs={"max_order": 3}),
        load_metric("bleu", compute_kwargs={"max_order": 4}),
        load_metric("rouge"),
        load_metric("bertscore", compute_kwargs={"lang": "tr"}),
    ],
}


class Evaluation:
    r"""
    Simple evaluation pipeline for text based metrics. By default it computes BLEU(n),
    METEOR, ROUGE-L and SacreBLEU metrics. It supports both QA and QG evaluation, when BMR metrics
    are given, it runs a QG Evaluation, for QA Evaluation construct the object with
    "squad".

    Note:

            If ``predictions`` and ``references`` are given as list of strings, the order is recieved
            as prediction & reference pairs and evaluation is done by prioratizing the order.
    """

    def __init__(self, metrics: Optional[List[str]] = None):
        self.metrics = metrics

    @staticmethod
    def task_to_metric(task: str) -> List[str]:
        return TASK_TO_METRIC.get(task)

    @staticmethod
    def metric_to_task(metric: str) -> str:
        for task, metrics in TASK_TO_METRIC.items():
            if metric in metrics:
                return task

    @staticmethod
    def get_tasks():
        return list(TASK_TO_METRIC.keys())

    @staticmethod
    def _get_task_related_samples(
        desired_task: str,
        predictions: Union[List[str], List[Dict]],
        references: Union[List[str], List[Dict]],
        tasks: Optional[List[str]] = None,
    ):
        if tasks is None:
            return predictions, references

        selected_predictions = []
        selected_references = []
        for prediction, reference, task in zip(predictions, references, tasks):
            if task == desired_task:
                selected_predictions.append(prediction)
                selected_references.append(reference)
        return selected_predictions, selected_references

    def run(
        self,
        predictions: Union[List[str], List[Dict]],
        references: Union[List[str], List[Dict]],
        tasks: List[str],
    ) -> Dict[str, Any]:
        scores = {}

        for task in TASK_TO_METRIC:
            metrics = self.task_to_metric(task)
            scorer = Jury(metrics=metrics, run_concurrent=True)
            selected_predictions, selected_references = self._get_task_related_samples(
                desired_task=task, predictions=predictions, references=references, tasks=tasks
            )
            save_json(
                {"predictions": selected_predictions, "references": selected_references},
                task + "_outputs_during_training.json",
            )
            task_scores = scorer(predictions=selected_predictions, references=selected_references)
            for key, value in task_scores.items():
                scores[task + "_" + key] = value

        return scores


def _get_qa_predictions_references(data: Dict) -> Tuple[List, List]:
    predictions = []
    references = []
    for article in data:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                predictions.append(qa["answer"])
                references.append(qa["gold_answer"])
    return predictions, references


def _get_qg_predictions_references(data: Dict) -> Tuple[List, List]:
    predictions = []
    references = []
    for article in data:
        for paragraph in article["paragraphs"]:
            for i, qa in enumerate(paragraph["qas"]):
                predictions.append(qa["question"])
                references.append(qa["gold_question"])
    return predictions, references


def _get_ans_ext_predictions_references(data: Dict, dont_combine_answer_list: bool = False) -> Tuple[List, List]:
    predictions = []
    references = []
    for article in data:
        for paragraph in article["paragraphs"]:
            if dont_combine_answer_list:
                predictions.append(paragraph["predicted_answer_list"])
                references.append(paragraph["gold_answer_list"])
            else:
                predictions.append(" ".join(paragraph["predicted_answer_list"]))
                references.append(" ".join(paragraph["gold_answer_list"]))
    return predictions, references


def evaluate_from_file(path: str, task: str, output: str = None, dont_combine_answer_list: bool = False):
    # prepare predictions & references
    data = load_json(path)["data"]
    if task == "qa":
        predictions, references = _get_qa_predictions_references(data)
    elif task == "qg":
        predictions, references = _get_qg_predictions_references(data)
    elif task == "ans_ext":
        predictions, references = _get_ans_ext_predictions_references(data, dont_combine_answer_list)
    else:
        raise ValueError("Unknown task. Must be one of [qa, qg, ans_ext]")
    # prepare metrics
    metrics = TASK_TO_METRIC.get(task)

    try:
        # calculate scores
        scorer = Jury(metrics=metrics, run_concurrent=False)
        scores = scorer(predictions=predictions, references=references)
        # log results
        logger.info(scores)
        # export result
        if output is None:
            export_path = str(Path(path).parent / (task + "_eval_scores.json"))
        else:
            export_path = output
        save_json(
            scores,
            export_path,
        )
        return scores
    except Exception as e:
        logger.warning(e)
        return None


def evaluate_on_train_end(model_args, training_args):
    logger.info("*** Evaluate on train end ***")

    if model_args.model_type == "mt5":
        eval_tasks = training_args.mt5_task_list
    elif model_args.model_type == "bert":
        eval_tasks = ["qa_for_bert"]

    overall_results = {}
    for eval_dataset in training_args.eval_dataset_list:
        for eval_task in eval_tasks:
            dataset_name = Path(eval_dataset).name
            data = load_dataset(eval_dataset)
            output_generation_file = os.path.join(
                training_args.output_dir, dataset_name + "_" + eval_task + "_generation.json"
            )
            logger.info(f"Evaluating on {dataset_name} for {eval_task} task.")

            if eval_task == "ans_ext":
                generate(
                    path_or_dict=data,
                    output=output_generation_file,
                    model_url_or_path=training_args.output_dir,
                    use_cuda=not training_args.no_cuda,
                    task=eval_task,
                    max_source_length=training_args.max_source_length,
                    max_target_length=training_args.max_target_length,
                    seed=training_args.seed,
                )
                output_eval_file = os.path.join(
                    training_args.output_dir, dataset_name + "_" + eval_task + "_eval_result.json"
                )
                results = evaluate_from_file(
                    path=output_generation_file, task=eval_task, output=output_eval_file, dont_combine_answer_list=False
                )
                if results is not None:
                    for key, value in results.items():
                        overall_results["eval_" + dataset_name + "_" + eval_task + "_" + key] = value
            elif eval_task == "qa":
                generate(
                    path_or_dict=data,
                    output=output_generation_file,
                    model_url_or_path=training_args.output_dir,
                    use_cuda=not training_args.no_cuda,
                    task=eval_task,
                    max_source_length=training_args.max_source_length,
                    max_target_length=training_args.max_target_length,
                    seed=training_args.seed,
                )
                output_eval_file = os.path.join(
                    training_args.output_dir, dataset_name + "_" + eval_task + "_eval_result.json"
                )
                results = evaluate_from_file(
                    path=output_generation_file,
                    task=eval_task,
                    output=output_eval_file,
                    dont_combine_answer_list=False,
                )
                for key, value in results.items():
                    overall_results["eval_" + dataset_name + "_" + eval_task + "_" + key] = value
            elif eval_task == "qg":
                generate(
                    path_or_dict=data,
                    output=output_generation_file,
                    model_url_or_path=training_args.output_dir,
                    use_cuda=not training_args.no_cuda,
                    task=eval_task,
                    use_answers=True,
                    max_source_length=training_args.max_source_length,
                    max_target_length=training_args.max_target_length,
                    qg_format=training_args.mt5_qg_format,
                    seed=training_args.seed,
                )
                output_eval_file = os.path.join(
                    training_args.output_dir, dataset_name + "_" + eval_task + "_eval_result.json"
                )
                results = evaluate_from_file(
                    path=output_generation_file,
                    task=eval_task,
                    output=output_eval_file,
                    dont_combine_answer_list=False,
                )
                for key, value in results.items():
                    overall_results["eval_" + dataset_name + "_" + eval_task + "_" + key] = value
            elif eval_task == "qa_for_bert":
                generate(
                    path_or_dict=data,
                    output=output_generation_file,
                    model_url_or_path=training_args.output_dir,
                    use_cuda=not training_args.no_cuda,
                    task=eval_task,
                    max_source_length=training_args.max_source_length,
                    max_target_length=training_args.max_target_length,
                    seed=training_args.seed,
                )
                output_eval_file = os.path.join(
                    training_args.output_dir, dataset_name + "_" + eval_task + "_eval_result.json"
                )
                results = evaluate_from_file(
                    path=output_generation_file,
                    task="qa",
                    output=output_eval_file,
                    dont_combine_answer_list=False,
                )
                for key, value in results.items():
                    overall_results["eval_" + dataset_name + "_" + "qa" + "_" + key] = value
    return overall_results
