import itertools
import logging
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from utils.nlp import sentence_tokenize


class QGPipeline:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        ans_model: PreTrainedModel,
        ans_tokenizer: PreTrainedTokenizer,
        qg_format: str,
        use_cuda: bool,
        generate_max_length: int = 80,
        generate_num_beams: int = 4,
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.ans_model = ans_model
        self.ans_tokenizer = ans_tokenizer

        self.qg_format = qg_format

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

        if self.ans_model is not self.model:
            self.ans_model.to(self.device)

        assert self.model.__class__.__name__ in ["MT5ForConditionalGeneration"]

        self.model_type = "mt5"

        self.generate_max_length = generate_max_length
        self.generate_num_beams = generate_num_beams

    def __call__(self, inputs: str):
        inputs = " ".join(inputs.split())
        sents, answers = self._extract_answers(inputs)
        flat_answers = list(itertools.chain(*answers))

        if len(flat_answers) == 0:
            return []

        qg_examples = self._prepare_inputs_for_qg_from_answers_hl(sents, answers)

        qg_inputs = [example["source_text"] for example in qg_examples]
        questions = self._generate_questions(qg_inputs)
        output = [{"answer": example["answer"], "question": que} for example, que in zip(qg_examples, questions)]
        return output

    def _generate_questions(self, inputs):
        inputs = self._tokenize(inputs, padding=True, truncation=True)

        outs = self.model.generate(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
            max_length=self.generate_max_length,
            num_beams=self.generate_num_beams,
        )

        questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        return questions

    def _extract_answers(self, context):
        sents, inputs = self._prepare_inputs_for_ans_extraction(context)

        inputs = self._tokenize(inputs, padding=True, truncation=True)

        outs = self.ans_model.generate(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
            max_length=self.generate_max_length,
        )

        dec = [self.ans_tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

        answers = [item.split("<sep>") for item in dec]

        answers = [i[:-1] for i in answers]

        return sents, answers

    def _tokenize(self, inputs, padding=True, truncation=True, add_special_tokens=True, max_length=512):
        inputs = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt",
        )
        return inputs

    def _prepare_inputs_for_ans_extraction(self, text):
        sents = sentence_tokenize(text)

        inputs = []
        for i in range(len(sents)):
            source_text = "extract answers:"
            for j, sent in enumerate(sents):
                if i == j:
                    sent = "<hl> %s <hl>" % sent
                source_text = "%s %s" % (source_text, sent)
                source_text = source_text.strip()

            # if self.model_type == "mt5":
            #    source_text = source_text + " </s>"
            inputs.append(source_text)

        return sents, inputs

    def _prepare_inputs_for_qg_from_answers_hl(self, sents, answers):
        inputs = []
        for i, answer in enumerate(answers):
            if len(answer) == 0:
                continue
            for answer_text in answer:
                sent = sents[i]
                sents_copy = sents[:]

                answer_text = answer_text.strip()
                # sometimes extracted answers does not match with the original text :/
                try:
                    ans_start_idx = sent.index(answer_text)

                    sent = f"{sent[:ans_start_idx]} <hl> {answer_text} <hl> {sent[ans_start_idx + len(answer_text): ]}"
                    sents_copy[i] = sent

                    source_text = " ".join(sents_copy)
                    source_text = f"generate question: {source_text}"
                    # if self.model_type == "mt5":
                    #    source_text = source_text + " </s>"
                except:
                    continue

                inputs.append({"answer": answer_text, "source_text": source_text})

        return inputs


class MultiTaskQAQGPipeline(QGPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, inputs: Union[Dict, str]):
        if type(inputs) is str:
            # do qg
            return super().__call__(inputs)
        else:
            # do qa
            return self._extract_answer(inputs["question"], inputs["context"])

    def _prepare_inputs_for_qa(self, question, context):
        source_text = f"question: {question}  context: {context}"
        # if self.model_type == "mt5":
        #    source_text = source_text + " </s>"
        return source_text

    def _extract_answer(self, question, context):
        source_text = self._prepare_inputs_for_qa(question, context)
        inputs = self._tokenize([source_text], padding=False)

        outs = self.model.generate(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
            max_length=self.generate_max_length,
        )

        answer = self.tokenizer.decode(outs[0], skip_special_tokens=True)

        return answer


SUPPORTED_TASKS = {
    "multitask-qa-qg": {
        "class": MultiTaskQAQGPipeline,
        "default": {
            "model": "obss/mt5-qa-qg",
        },
    },
}


def pipeline(
    task: str,
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    qg_format: Optional[str] = "highlight",
    use_cuda: Optional[bool] = True,
    **kwargs,
):
    # Retrieve the task
    if task not in SUPPORTED_TASKS:
        raise KeyError("Unknown task {}, available tasks are {}".format(task, list(SUPPORTED_TASKS.keys())))

    targeted_task = SUPPORTED_TASKS[task]
    task_class = targeted_task["class"]

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        model = targeted_task["default"]["model"]

    # Try to infer tokenizer from model or config name (if provided as str)
    if tokenizer is None:
        if isinstance(model, str):
            tokenizer = model
        else:
            # Impossible to guest what is the right tokenizer here
            raise Exception(
                "Impossible to guess which tokenizer to use. "
                "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
            )

    # Instantiate tokenizer if needed
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    # Instantiate model if needed
    if isinstance(model, str):
        model = AutoModelForSeq2SeqLM.from_pretrained(model)

    return task_class(
        model=model,
        tokenizer=tokenizer,
        ans_model=model,
        ans_tokenizer=tokenizer,
        qg_format=qg_format,
        use_cuda=use_cuda,
    )
