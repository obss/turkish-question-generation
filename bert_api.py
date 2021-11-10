import logging
from typing import Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from hf.model import BertModel
from utils.file import load_json

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def _postprocess_output(
    model_output, top_n_best_answers: int = 20, max_answer_length: int = 64
) -> Tuple[Union[int, int]]:
    """returns valid_answer_list (List[Dict]) with each elements having
    score (float), answer_start (int), answer_end (int) keys"""

    start_logits = model_output.start_logits[0].cpu().detach().numpy()
    end_logits = model_output.end_logits[0].cpu().detach().numpy()
    # Gather the indices the best start/end logits:
    start_indexes = np.argsort(start_logits)[-1 : -top_n_best_answers - 1 : -1].tolist()
    end_indexes = np.argsort(end_logits)[-1 : -top_n_best_answers - 1 : -1].tolist()
    valid_answers = []
    for start_index in start_indexes:
        for end_index in end_indexes:
            # Don't consider answers with a length that is either < 0 or > max_answer_length.
            if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                continue
            valid_answers.append(
                {
                    "score": start_logits[start_index] + end_logits[end_index],
                    "answer_start": start_index,
                    "answer_end": end_index,
                }
            )
    valid_answer_list = sorted(valid_answers, key=lambda x: x["score"], reverse=True)
    return valid_answer_list


def qa_from_file(
    path_or_dict: str,
    model_url_or_path: str,
    use_cuda: bool = True,
    top_n_best_answers: int = 20,
    max_source_lenght: int = 512,
    max_target_length: int = 80,
):
    model = BertModel(model_url_or_path, use_cuda=use_cuda)

    # read data from path or dict
    if isinstance(path_or_dict, str):
        data = load_json(path_or_dict)["data"]
    else:
        data = path_or_dict["data"]

    out = {"data": []}
    for article in tqdm(data, desc="Answer Generation using Bert from articles"):
        out_article = {"paragraphs": [], "title": article.get("title")}
        # iterate over each paragraph
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            out_para = {"context": context, "qas": []}

            # extract questions from dataset paragraph and get answer from model
            for qa in paragraph["qas"]:
                question = qa.get("question")
                if question is not None:

                    inputs = model.tokenizer.encode_plus(
                        context,
                        question,
                        max_length=max_source_lenght,
                        add_special_tokens=True,
                        padding=False,
                        pad_to_max_length=False,
                        return_tensors="pt",
                        truncation=True,
                    )
                    input_ids = inputs["input_ids"].tolist()[0]

                    model_output = model.model(torch.tensor([input_ids]).to(model.device))
                    # answer_dict_list = _postprocess_output(
                    #    model_output, top_n_best_answers=top_n_best_answers, max_answer_length=max_target_length
                    # )
                    answer_start_scores, answer_end_scores = model_output["start_logits"], model_output["end_logits"]

                    answer_start = torch.argmax(
                        answer_start_scores
                    )  # Get the most likely beginning of answer with the argmax of the score
                    answer_end = (
                        torch.argmax(answer_end_scores) + 1
                    )  # Get the most likely end of answer with the argmax of the score

                    # answer_list = []
                    # for answer_dict in answer_dict_list:
                    #    answer = model.tokenizer.convert_tokens_to_string(
                    #        model.tokenizer.convert_ids_to_tokens(
                    #            input_ids[answer_dict["answer_start"] : answer_dict["answer_end"]]
                    #        )
                    #    )
                    #    answer_list.append(answer)
                    # append q&a pair into out_para
                    if answer_end > answer_start:
                        answer = model.tokenizer.convert_tokens_to_string(
                            model.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
                        )
                    else:
                        answer = ""
                    qa_out = {
                        "answer": answer,
                        # "alternative_answers": answer_list[1:],
                        "gold_answer": qa["answers"][0]["text"],
                        "question": question,
                    }
                    out_para["qas"].append(qa_out)
                else:
                    logger.warning("skipping a paragraph without questions.")

            out_article["paragraphs"].append(out_para)
        out["data"].append(out_article)
    return out
