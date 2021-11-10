import itertools
import logging
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Union

from tqdm import tqdm

from hf.model import MT5Model
from utils.file import load_json
from utils.nlp import (
    add_start_end_to_answer_list_per_sentence,
    normalize_text,
    postprocess_answer_extraction_output,
    prepare_answer_extraction_samples,
    prepare_qa_sample,
    prepare_qg_samples,
    sentence_tokenize,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class TurQue:
    def __init__(
        self,
        model_url_or_path: str = None,
        use_cuda: bool = None,
        max_source_length: int = 512,
        max_target_length: int = 80,
        generate_num_beams: int = 4,
        top_k: int = None,
        top_p: float = None,
        qg_format: str = "highlight",
    ):
        model_url_or_path = "turque-s1" if model_url_or_path is None else model_url_or_path
        mt5_model = MT5Model(model_url_or_path, use_cuda=use_cuda)
        self.model = mt5_model.model
        self.tokenizer = mt5_model.tokenizer
        self.model_type = mt5_model.type

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.generate_num_beams = generate_num_beams
        self.top_k = top_k
        self.top_p = top_p
        self.qg_format = qg_format

    def __call__(
        self,
        task: str,
        context: str,
        question: Optional[str] = None,
        answer_list: Optional[Union[List[str], List[Dict]]] = None,
    ):
        context = normalize_text(context)
        if task == "answer-extraction":
            # return answer list using context
            answer_list = self._generate_answer_list_from_context(context)
            output = [{"answer": answer["text"]} for answer in answer_list]
            return output
        elif task == "question-answering":
            # return answer list using context and question
            question = normalize_text(question)
            answer = self._generate_answer_from_context_and_question(question, context)
            output = [{"answer": answer}]
            return output
        elif task == "question-generation":
            # return question list using context
            if answer_list is None:
                answer_list = self._generate_answer_list_from_context(context)

                if not answer_list:
                    return [{"answer": None, "question": None}]
            else:
                _answer_list = []
                for answer in answer_list:
                    answer["text"] = normalize_text(answer["text"])
                    _answer_list.append(answer)
                answer_list = _answer_list

            samples = prepare_qg_samples(context, answer_list, qg_format=self.qg_format)

            if samples[0]["answer"] is None:
                return [{"answer": None, "question": None}]
            else:
                inputs = [sample["source_text"] for sample in samples]

                # single generation without padding is 5 times faster than padding + batch generation
                question_list = []
                for input in inputs:
                    question = self._generate_question_list([input], padding=False)[0]
                    question_list.append(question)

                output = [
                    {"answer": sample["answer"], "question": question}
                    for sample, question in zip(samples, question_list)
                ]
                return output
        else:
            raise NameError(
                f"{task} is not defined. 'task' must be one of ['answer-extraction', 'question-answering', 'question-generation']"
            )

    def _generate_question_list(self, inputs, padding=True, truncation=True):
        inputs = self._tokenize(inputs, padding=padding, truncation=truncation)

        outs = self.model.generate(
            input_ids=inputs["input_ids"].to(self.model.device),
            attention_mask=inputs["attention_mask"].to(self.model.device),
            max_length=self.max_target_length,
            num_beams=self.generate_num_beams,
            top_k=self.top_k,
            top_p=self.top_p,
        )

        question_list = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        return question_list

    def _generate_answer_list_from_context(self, context):
        samples = prepare_answer_extraction_samples(context=context)
        inputs = [sample["source_text"] for sample in samples]

        inputs = self._tokenize(inputs, padding=True, truncation=True)

        outs = self.model.generate(
            input_ids=inputs["input_ids"].to(self.model.device),
            attention_mask=inputs["attention_mask"].to(self.model.device),
            max_length=self.max_target_length,
        )

        output_list = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

        answer_list_per_sentence = []
        for output in output_list:
            # postprocess answer extraction output list
            answer_text_list = postprocess_answer_extraction_output(output)
            answer_list = [{"text": normalize_text(answer_text)} for answer_text in answer_text_list]
            answer_list_per_sentence.append(answer_list)

        sentence_list = sentence_tokenize(context)
        answer_list = add_start_end_to_answer_list_per_sentence(sentence_list, answer_list_per_sentence)

        return answer_list

    def _tokenize(self, inputs, padding=True, truncation=True, add_special_tokens=True):
        inputs = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=self.max_source_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt",
        )
        return inputs

    def _generate_answer_from_context_and_question(self, question, context):
        sample = prepare_qa_sample(context, question)
        source_text = sample["source_text"]
        inputs = self._tokenize([source_text], padding=False)

        outs = self.model.generate(
            input_ids=inputs["input_ids"].to(self.model.device),
            attention_mask=inputs["attention_mask"].to(self.model.device),
            max_length=self.max_target_length,
        )

        answer = self.tokenizer.decode(outs[0], skip_special_tokens=True)

        return answer

    def qg_from_file(self, path_or_dict: str, use_answers: bool = False, **kwargs):
        """performs question-generation using the contexts and answers
        from squad formatted json file or answers extracted by the model"""

        for k, v in kwargs.items():
            setattr(self, k, v)
        task = "question-generation"

        # read data from path or dict
        if isinstance(path_or_dict, str):
            data = load_json(path_or_dict)["data"]
        else:
            data = path_or_dict["data"]

        out = {"data": []}
        for article in tqdm(data, desc="Question Generation from articles"):
            out_article = {"paragraphs": [], "title": article.get("title")}
            # iterate over each paragraph
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                out_para = {"context": context, "qas": []}
                if use_answers and paragraph["qas"]:
                    answer_list = []
                    paragraph["qas"] = sorted(paragraph["qas"], key=lambda k: k["answers"][0]["answer_start"])
                    for qa in paragraph["qas"]:
                        answer = qa["answers"][0]
                        answer_list.append(answer)
                elif use_answers and not paragraph["qas"]:  # pass if paragraph["qas"] is empty
                    continue
                else:
                    answer_list = None
                qg_out = self(task=task, context=context, answer_list=answer_list)
                if qg_out[0]["question"] is None:
                    continue
                for qa, gold_qa in zip(qg_out, paragraph["qas"]):
                    if use_answers:
                        qa["gold_question"] = gold_qa["question"]
                    out_para["qas"].append(qa)
                out_article["paragraphs"].append(out_para)
            out["data"].append(out_article)
        return out

    def qa_from_file(self, path_or_dict: str, **kwargs):
        """performs question-answering using the contexts and questions
        from squad formatted json file"""

        for k, v in kwargs.items():
            setattr(self, k, v)
        task = "question-answering"

        # read data from path or dict
        if isinstance(path_or_dict, str):
            data = load_json(path_or_dict)["data"]
        else:
            data = path_or_dict["data"]

        out = {"data": []}
        for article in tqdm(data, desc="Question Answering from articles"):
            out_article = {"paragraphs": [], "title": article.get("title")}
            # iterate over each paragraph
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                out_para = {"context": context, "qas": []}

                # extract questions from dataset paragraph and get answer from model
                for qa in paragraph["qas"]:
                    question = qa.get("question")
                    if question is not None:
                        qa_out = self(task=task, context=context, question=question)[0]
                        # append q&a pair into out_para
                        qa_out["gold_answer"] = qa["answers"][0]["text"]
                        qa_out["question"] = question
                        out_para["qas"].append(qa_out)
                    else:
                        logger.warning("skipping a paragraph without questions.")

                out_article["paragraphs"].append(out_para)
            out["data"].append(out_article)
        return out

    def ans_ext_from_file(self, path_or_dict: str, **kwargs):
        """performs answer-extraction using the contexts from squad formatted json file"""

        for k, v in kwargs.items():
            setattr(self, k, v)
        task = "answer-extraction"

        # read data from path or dict
        if isinstance(path_or_dict, str):
            data = load_json(path_or_dict)["data"]
        else:
            data = path_or_dict["data"]

        out = {"data": []}
        for article in tqdm(data, desc="Answer Extraction from articles"):
            out_article = {"paragraphs": [], "title": article.get("title")}
            # iterate over each paragraph
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                out_para = {"context": context, "gold_answer_list": [], "predicted_answer_list": []}

                if paragraph["qas"]:
                    gold_answer_list = []
                    paragraph["qas"] = sorted(paragraph["qas"], key=lambda k: k["answers"][0]["answer_start"])
                    for qa in paragraph["qas"]:
                        answer = qa["answers"][0]
                        gold_answer_list.append(answer["text"])
                else:
                    logger.warning("skipping a paragraph without q/a's.")
                # extract answers
                ans_ext_out = self(task=task, context=context)
                # add gold and predicted answers
                predicted_answer_list = [output["answer"] for output in ans_ext_out]
                out_para["gold_answer_list"] = gold_answer_list
                out_para["predicted_answer_list"] = predicted_answer_list

                out_article["paragraphs"].append(out_para)
            out["data"].append(out_article)
        return out
