import json
from typing import Optional, Union

from transformers import set_seed

from core.api import TurQue
from core.bert_api import qa_from_file
from utils.file import save_json


def read_config(path: str):
    with open(path, "r", encoding="utf-8") as jf:
        config = json.load(jf)
    return config


def generate_qg(
    path_or_dict: str = None,
    output: str = None,
    model_url_or_path: str = None,
    use_cuda: bool = True,
    use_answers: str = None,
    max_source_length: int = 512,
    max_target_length: int = 80,
    qg_format: str = "highlight",
):
    use_answers = False if use_answers is None else use_answers
    turque = TurQue(
        model_url_or_path=model_url_or_path,
        use_cuda=use_cuda,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        qg_format=qg_format,
    )
    result = turque.qg_from_file(path_or_dict=path_or_dict, use_answers=use_answers)
    save_json(result, output)


def generate_qa(
    path_or_dict: str = None,
    output: str = None,
    model_url_or_path: str = None,
    use_cuda: bool = True,
    max_source_length: int = 512,
    max_target_length: int = 80,
):
    turque = TurQue(
        model_url_or_path=model_url_or_path,
        use_cuda=use_cuda,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    result = turque.qa_from_file(path_or_dict=path_or_dict)
    save_json(result, output)


def generate_ans_ext(
    path_or_dict: str = None,
    output: str = None,
    model_url_or_path: str = None,
    use_cuda: bool = True,
    max_source_length: int = 512,
    max_target_length: int = 80,
):
    turque = TurQue(
        model_url_or_path=model_url_or_path,
        use_cuda=use_cuda,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    result = turque.ans_ext_from_file(path_or_dict=path_or_dict)
    save_json(result, output)


def generate_qa_for_bert(
    path_or_dict: str,
    output: str,
    model_url_or_path: str = None,
    use_cuda: bool = True,
    max_source_length: int = 512,
    max_target_length: int = 80,
):
    result = qa_from_file(
        path_or_dict=path_or_dict,
        model_url_or_path=model_url_or_path,
        use_cuda=use_cuda,
        max_source_lenght=max_source_length,
        max_target_length=max_target_length,
    )
    save_json(result, output)


def generate(
    path_or_dict: str = None,
    output: str = None,
    model_url_or_path: str = None,
    use_cuda: bool = True,
    use_answers: Union[str, bool] = False,
    task: str = "qa",
    max_source_length: int = 512,
    max_target_length: int = 80,
    config: str = None,
    qg_format: str = "highlight",
    seed: Optional[int] = None,
):
    """
    path_or_dict (str): path or dict for a squad formatted dataset
    output (str): output path for generation json
    use_cuda (bool): perform generation on cuda
    use_answers (bool): use gold answer for qg
    task (str): one of 'qa', 'qg', 'ans_ext', 'qa_for_bert'
    config (str): path to a json file
    qg_format (str): 'highlight', 'prepend' or 'both'
    seed (int): seed for randomized operations
    """
    args = read_config(config) if config is not None else {}
    path_or_dict = args.get("path_or_dict") if path_or_dict is None else path_or_dict
    output = args.get("output") if output is None else output
    model_url_or_path = args.get("model_url_or_path") if model_url_or_path is None else model_url_or_path
    task = args.get("task") if task is None else task
    use_answers = False if use_answers is None else use_answers
    if seed is not None:
        set_seed(seed)
    if task == "qa":
        generate_qa(
            path_or_dict=path_or_dict,
            output=output,
            model_url_or_path=model_url_or_path,
            use_cuda=use_cuda,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
        )
    elif task == "qg":
        generate_qg(
            path_or_dict=path_or_dict,
            output=output,
            model_url_or_path=model_url_or_path,
            use_cuda=use_cuda,
            use_answers=use_answers,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            qg_format=qg_format,
        )
    elif task == "ans_ext":
        generate_ans_ext(
            path_or_dict=path_or_dict,
            output=output,
            model_url_or_path=model_url_or_path,
            use_cuda=use_cuda,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
        )
    elif task == "qa_for_bert":
        generate_qa_for_bert(
            path_or_dict=path_or_dict,
            output=output,
            model_url_or_path=model_url_or_path,
            use_cuda=use_cuda,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
        )
    else:
        raise ValueError(f"'task' should be one of ['qa', 'qg', 'ans_ext', 'qa_for_bert'] but given as {task}")
