import os
from typing import Dict, List

from datasets import Dataset

from utils.file import load_json
from utils.nlp import (
    download_and_load_dataset,
    normalize_text,
    prepare_answer_extraction_samples,
    prepare_qa_sample,
    prepare_qg_samples,
)

# former url train data was unfit for (pyarrow) datasets.Dataset (file maybe corrupt).

_TQUAD_URL = "https://github.com/fcakyon/turkish-qa-datasets/releases/download/0.0.1/"
_TQUAD1_DEV_FILE = "tquad_dev_data_v1.json"
_TQUAD1_TRAINING_FILE = "tquad_train_data_v1.json"
_TQUAD2_DEV_FILE = "tquad_dev_data_v2.json"
_TQUAD2_TRAINING_FILE = "tquad_train_data_v2.json"
_TQUAD_DEV_SMALL_FILE = "tquad_dev_small_data.json"
_TQUAD_LOCAL_DIR = "data/tquad/"
_XQUAD_TR_URL = "https://raw.githubusercontent.com/deepmind/xquad/master/"
_XQUAD_LOCAL_DIR = "data/xquad"


def load_tquad_data(data_name="tquad2-train") -> Dataset:
    if "tquad1-train" in data_name:
        tquad_url = os.path.join(_TQUAD_URL, _TQUAD1_TRAINING_FILE)
        tquad_local_path = os.path.join(_TQUAD_LOCAL_DIR, _TQUAD1_TRAINING_FILE)
    elif "tquad2-train" in data_name:
        tquad_url = os.path.join(_TQUAD_URL, _TQUAD2_TRAINING_FILE)
        tquad_local_path = os.path.join(_TQUAD_LOCAL_DIR, _TQUAD2_TRAINING_FILE)
    elif "tquad1-valid" in data_name:
        tquad_url = os.path.join(_TQUAD_URL, _TQUAD1_DEV_FILE)
        tquad_local_path = os.path.join(_TQUAD_LOCAL_DIR, _TQUAD1_DEV_FILE)
    elif "tquad2-valid" in data_name:
        tquad_url = os.path.join(_TQUAD_URL, _TQUAD2_DEV_FILE)
        tquad_local_path = os.path.join(_TQUAD_LOCAL_DIR, _TQUAD2_DEV_FILE)
    elif "small" in data_name:
        tquad_url = os.path.join(_TQUAD_URL, _TQUAD_DEV_SMALL_FILE)
        tquad_local_path = os.path.join(_TQUAD_LOCAL_DIR, _TQUAD_DEV_SMALL_FILE)
    else:
        raise ValueError(
            f"Unknown data_name {data_name}, must be one of ['tquad1-train', 'tquad2-train', 'tquad1-valid', 'tquad2-valid', 'tquad.small']"
        )

    return download_and_load_dataset(tquad_url, tquad_local_path)


def load_xquad_data(data_name="xquad.tr"):
    """XQuad dataset has only validation split."""
    xquad_url = os.path.join(_XQUAD_TR_URL, data_name + ".json")
    xquad_local_path = os.path.join(_XQUAD_LOCAL_DIR, data_name + ".json")
    return download_and_load_dataset(xquad_url, xquad_local_path)


def prepare_data_for_bert(data: Dataset) -> List[Dict]:
    """
    Args:
        data: squad data

    Returns: Processed samples as list in bert input format.
    """
    samples = []
    data = data["data"]

    for group in data:
        for passage in group["paragraphs"]:
            context = passage["context"]
            if passage["qas"]:
                for qa in passage["qas"]:
                    question = qa["question"]
                    # for answer in qa["answers"]:
                    answer = qa["answers"][0]

                    gold_text = answer["text"]
                    start_idx = answer["answer_start"]
                    end_idx = start_idx + len(gold_text)

                    # sometimes squad answers are off by a character or two – fix this
                    if context[start_idx:end_idx] == gold_text:
                        answer["answer_end"] = end_idx
                    elif context[start_idx - 1 : end_idx - 1] == gold_text:
                        answer["answer_start"] = start_idx - 1
                        answer["answer_end"] = end_idx - 1  # When the gold label is off by one character
                    elif context[start_idx - 2 : end_idx - 2] == gold_text:
                        answer["answer_start"] = start_idx - 2
                        answer["answer_end"] = end_idx - 2  # When the gold label is off by two characters
                    elif context[start_idx - 3 : end_idx - 3] == gold_text:
                        answer["answer_start"] = start_idx - 3
                        answer["answer_end"] = end_idx - 3  # When the gold label is off by three characters
                    else:
                        print(
                            f"skipping the answer|answer_start|context {answer['text']}|{answer['answer_start']}|{context} | for reason: 'answer indexes are off by a lot'"
                        )
                        continue

                    sample = {"context": context, "question": question, "answer": answer}
                    samples.append(sample)
    return samples


def prepare_data_for_mt5(
    data: Dataset, task_list: List[str] = ["ans_ext", "qa", "qg"], qg_format="highlight"
) -> List[Dict]:
    """
    Args:
        data: squad data
        task_list: list of tasks to data be prepared
        qg_format: "highlight", "prepend" or "both"

    Returns: Processed samples as list in mt5 input format.
    """
    samples = []
    data = data["data"]

    for article in data:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            question_list = []
            answer_list = []

            if paragraph["qas"]:  # pass if paragraph["qas"] is empty
                for qa in paragraph["qas"]:
                    question = normalize_text(qa["question"])
                    answer = qa["answers"][0]
                    answer["text"] = answer["text"]
                    qa_sample = prepare_qa_sample(context=context, question=question, answer=answer["text"])
                    if qa_sample["target_text"] is not None:
                        if "qa" in task_list:
                            samples.append(qa_sample)
                        question_list.append(question)
                        answer_list.append(answer)

                if answer_list and question_list:
                    qg_samples = prepare_qg_samples(
                        context=context, answer_list=answer_list, question_list=question_list, qg_format=qg_format
                    )
                    if qg_samples[0]["answer"] is not None:
                        if "qg" in task_list:
                            samples.extend(qg_samples)

                answer_extraction_samples = prepare_answer_extraction_samples(context=context, answer_list=answer_list)
                for answer_extraction_sample in answer_extraction_samples:
                    if answer_extraction_sample["target_text"] is not None:
                        if "ans_ext" in task_list:
                            samples.extend(answer_extraction_samples)
    return samples


def prepare_data(
    data: Dict, target_format="mt5", mt5_task_list: List[str] = ["ans_ext", "qa", "qg"], mt5_qg_format="highlight"
):
    """
    Args:
        target_format (str): output format ('mt5' or 'bert')
        mt5_task_list: list of tasks for mt5 data to be prepared
        mt5_qg_format: "highlight", "prepend" or "both"
    """
    if target_format == "mt5":
        samples = prepare_data_for_mt5(data, mt5_task_list, mt5_qg_format)
    elif target_format == "bert":
        samples = prepare_data_for_bert(data)
    return samples


def load_dataset(name_or_path: str):
    if os.path.isfile(name_or_path):
        data = load_json(name_or_path)
    elif "tquad" in name_or_path:
        data = load_tquad_data(name_or_path)
    elif "xquad" in name_or_path:
        data = load_xquad_data(name_or_path)
    else:
        raise ValueError(f"Unknown dataset {name_or_path}.")

    return data


def load_and_prepare_dataset(
    name_or_path: str,
    target_format="mt5",
    mt5_task_list: List[str] = ["ans_ext", "qa", "qg"],
    mt5_qg_format="highlight",
):
    """
    Args:
        target_format (str): output format ('mt5' or 'bert')
        mt5_task_list: list of tasks for mt5 data to be prepared
        mt5_qg_format: "highlight", "prepend" or "both"
    """
    data = load_dataset(name_or_path)
    return prepare_data(data, target_format=target_format, mt5_task_list=mt5_task_list, mt5_qg_format=mt5_qg_format)
