import itertools
import re
import string
from typing import Dict, List, Union

from datasets import Dataset
from trtokenizer.tr_tokenizer import SentenceTokenizer

from utils import SOURCE_ROOT
from utils.file import attempt_download, load_json


def sentence_tokenize(context: str) -> List[str]:
    non_prefix_file_path = str(SOURCE_ROOT / "tr_non_suffixes")
    sentence_tokenizer = SentenceTokenizer(non_breaking_prefix_file=non_prefix_file_path)
    context = context.replace("\xa0", " ").replace("\ufeff", " ").replace("\t", " ")

    sentence_list = []
    for trtok_sentence in sentence_tokenizer.tokenize(context):

        pattern = re.escape(trtok_sentence)
        pattern = " +".join(pattern.split())
        pattern += " {0,1}"  # handle space between sentences
        pattern = r"%s" % pattern
        pattern += "\r{0,1}"  # handle \r between sentences
        pattern = r"%s" % pattern
        pattern += "\n{0,1}"  # handle \n between sentences
        pattern = r"%s" % pattern
        match_str = re.search(pattern, context)
        start_idx, end_idx = match_str.span()
        sentence = context[start_idx:end_idx]
        sentence_list.append(sentence)
    return sentence_list


def _get_correct_alignement(context, answer):
    """Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here."""
    gold_text = answer["text"]
    start_idx = answer["answer_start"]
    end_idx = start_idx + len(gold_text)
    if context[start_idx:end_idx] == gold_text:
        return start_idx, end_idx  # When the gold label position is good
    elif context[start_idx - 1 : end_idx - 1] == gold_text:
        return start_idx - 1, end_idx - 1  # When the gold label is off by one character
    elif context[start_idx - 2 : end_idx - 2] == gold_text:
        return start_idx - 2, end_idx - 2  # When the gold label is off by two character
    elif context[start_idx + 1 : end_idx + 1] == gold_text:
        return start_idx + 1, end_idx + 1  # When the gold label is off by one character
    elif context[start_idx + 2 : end_idx + 2] == gold_text:
        return start_idx + 2, end_idx + 2  # When the gold label is off by two character
    else:
        raise ValueError()


def get_answer(qa: Dict):
    """qa: each element of 'qas' field of squad formatted dataset paragraph"""
    if qa.get("answer") is not None:
        return qa["answer"]
    try:
        answer = qa["answers"][0]
    except IndexError:
        answer = None
    return answer


def get_answer_indices_from_sentence(answer_text: str, sentence: str, loose_match: bool = False):
    """
    first match is returned in match case

    Args:
        answer_text (str)
        sentence (str)
        loose_match (bool) : if True, regex also matches substrings

    Returns:
        None, if no match
        {"text": answer_text, "answer_start": ans_start_idx, "answer_end": ans_end_idx}, if match
            answer_start and answer_end are sentence-wise indexes, not context-wise
    """
    # sometimes extracted answers does not match with the original text :/
    try:
        pattern = r"(?<![a-zA-Z0-9])(%s)(?![a-zA-Z0-9])" % re.escape(answer_text)
        match_str = re.search(pattern, sentence)
        if match_str is None:
            # handle trailing whitespace at start/end
            answer_text = answer_text.strip()
            pattern = r"(?<![a-zA-Z0-9])(%s)(?![a-zA-Z0-9])" % re.escape(answer_text)
            match_str = re.search(pattern, sentence)
            if loose_match and match_str is None:
                # loose match
                pattern = r"%s" % re.escape(answer_text)
                match_str = re.search(pattern, sentence)
        ans_start_idx, ans_end_idx = match_str.span()
    except AttributeError as e:
        # No match
        if loose_match:
            print(f"skipping the answer|sentence pair: {answer_text}| {sentence}| for reason: {e}")
        return None
    except Exception as e:
        # Incorrect structure
        if loose_match:
            print(f"skipping the answer|sentence pair: {answer_text}| {sentence}| for reason: {e}")
        return None
    else:
        answer = {"text": answer_text, "answer_start": ans_start_idx, "answer_end": ans_end_idx}
        return answer


def get_sentence_indices_list(sentence_list: List[str]):
    """
    Args:
        sentence_list List[str]

    Returns:
        List of 'start' and 'end' indexes of sentences
    """

    sentence_indices_list: List[Dict] = []
    prev_end_idx = 0
    for sentence_ind, sentence in enumerate(sentence_list):
        if sentence_ind == 0:
            start_idx = 0
            end_idx = len(sentence)
        else:
            start_idx = prev_end_idx
            end_idx = prev_end_idx + len(sentence)
        prev_end_idx = end_idx
        sentence_indices_list.append({"sentence_start": start_idx, "sentence_end": end_idx})
    return sentence_indices_list


def add_start_end_to_answer_list_per_sentence(sentence_list: List[str], answer_list_per_sentence: List[List[Dict]]):
    """
    Args:

    """
    sentence_indices_list = get_sentence_indices_list(sentence_list)

    _answer_list_per_sentence = []
    answer_text_list = []
    for ind, answer_list in enumerate(answer_list_per_sentence):
        _answer_list = []
        for answer in answer_list:
            # normalize text
            answer_text = normalize_text(answer["text"])
            # append if not present
            if answer_text and (answer_text not in answer_text_list):
                sentence_indices = sentence_indices_list[ind]
                sentence = sentence_list[ind]

                answer_text_list.append(answer_text)
                answer = get_answer_indices_from_sentence(answer_text, sentence)

                if answer is not None:
                    # convert sentence-wise start/end to context-wise start/end
                    answer["answer_start"] += sentence_indices["sentence_start"]
                    answer["answer_end"] += sentence_indices["sentence_start"]

                    _answer_list.append(answer)

        _answer_list_per_sentence.append(_answer_list)
    answer_list = list(itertools.chain(*_answer_list_per_sentence))
    return answer_list


def get_answer_list_per_sentence(
    sentence_list: List[str], answer_list: Union[List[Dict], List[List[Dict]]], question_list: List[str] = None
):
    """
    Args:
        sentence_list: List[str]
        answer_list:
        [
            {'text': str, 'answer_start': int},
            {'text': str, 'answer_start': int},
            ...
        ]
        or
        answer_list:
        [
            [{'text': str, 'answer_start': int}, {'text': str, 'answer_start': int}],
            [{'text': str, 'answer_start': int}],
            ...
        ]
        question_list: List[str] (optional)

    Returns:
        answer_list_per_sentence: [
            [
                {'text': str, 'answer_start': int, 'answer_end': int},
                {'text': str, 'answer_start': int, 'answer_end': int},
            ],
            [
                {'text': str, 'answer_start': int, 'answer_end': int},
                {'text': str, 'answer_start': int, 'answer_end': int},
            ],
            ...
        ]
        question_list_per_sentence: [
            str,
            str,
            ...
        ]
    """
    # split into sentences
    num_sentences = len(sentence_list)
    num_answers = len(answer_list)

    if question_list:
        num_questions = len(question_list)
        if not num_answers == num_questions:
            raise ValueError(
                f"'answer_list' ({num_answers}) and 'question_list' ({num_questions}) should have the same length"
            )

    # get positions of the sentences
    sentence_indices_list = get_sentence_indices_list(sentence_list)

    # convert answer_list_per_sentence to answer_list
    if isinstance(answer_list[0], list):
        answer_list = add_start_end_to_answer_list_per_sentence(sentence_list, answer_list)

    # get list of answers (and questions) for each sentence
    answer_list_per_sentence = []
    question_list_per_sentence = []
    for sentence_ind in range(num_sentences):
        sentence_indices = sentence_indices_list[sentence_ind]
        target_answers = []
        target_questions = []
        for answer_ind in range(num_answers):
            answer = answer_list[answer_ind]
            if answer["answer_start"] in range(
                sentence_indices["sentence_start"] - len(answer["text"]) + 1,
                sentence_indices["sentence_end"] - len(answer["text"]) + 1,
            ):  # +1s are required because of range()
                answer_text = answer["text"]
                sentence = sentence_list[sentence_ind]

                answer = get_answer_indices_from_sentence(answer_text, sentence, loose_match=True)
                if answer is None:
                    continue
                # convert sentence-wise start/end to context-wise start/end
                answer["answer_start"] += sentence_indices["sentence_start"]
                answer["answer_end"] += sentence_indices["sentence_start"]

                target_answers.append(answer)
                if question_list:
                    question = question_list[answer_ind]
                    target_questions.append(question.strip())

        answer_list_per_sentence.append(target_answers)
        if question_list:
            question_list_per_sentence.append(target_questions)

    if question_list:
        return answer_list_per_sentence, question_list_per_sentence
    else:
        return answer_list_per_sentence


def prepare_answer_extraction_samples(context: str, answer_list: List[Dict] = None):
    """
    Args:
        context: str (assumed to be normalized via normalize_text)
        answer_list: [
            {'text': str, 'answer_start': int},
            {'text': str, 'answer_start': int},
            ...
        ]
    """
    prepare_target = True if answer_list else False

    # split into sentences
    sentence_list = sentence_tokenize(context)
    num_sentences = len(sentence_list)

    if prepare_target:
        answer_list_per_sentence = get_answer_list_per_sentence(sentence_list, answer_list)

    # prepare sources (and targets)
    samples = []
    for sentence_ind in range(num_sentences):
        source_text = "extract answers:"

        if prepare_target:
            answer_list = answer_list_per_sentence[sentence_ind]
            answer_list = [answer["text"] for answer in answer_list]
            if not answer_list:
                continue
            answer_list = list(dict.fromkeys(answer_list))  # remove duplicate answers without changing the order
            target_text = " <sep> ".join(answer_list) + " <sep>"
        else:
            target_text = None

        for sentence_ind2, sentence in enumerate(sentence_list):
            if sentence_ind == sentence_ind2:
                sentence = f"<hl> {sentence} <hl>"
            source_text = f"{source_text} {sentence}"
            source_text = source_text.strip()

        sample = {"source_text": source_text, "target_text": target_text, "answer_list": answer_list}
        if sample["target_text"] is None:
            sample
        samples.append(sample)

    return samples


def prepare_qa_sample(context: str, question: str, answer: str = None):
    """
    Args:
        context (str) (assumed to be normalized via normalize_text)
        question (str)
        answer (str)
    """
    prepare_target = True if answer else False

    source_text = f"question: {question} context: {context}"
    if prepare_target:
        target_text = f"{answer}"
    else:
        target_text = None
    return {"source_text": source_text, "target_text": target_text}


def prepare_qg_samples(context, answer_list: List[Dict], question_list: List[str] = None, qg_format: str = "highlight"):
    """
    Args:
        context (str)
        question_list (List[str])
        answer_list: [
            {'text': str, 'answer_start': int},
            {'text': str, 'answer_start': int},
            ...
        ]
        qg_format: 'highlight', 'prepend' or 'both'
    """
    # split into sentences

    try:
        samples = []
        for ind, answer in enumerate(answer_list):
            start_pos, end_pos = _get_correct_alignement(context, answer)
            answer_text = answer["text"]

            if qg_format == "prepend":
                source_text = f"answer: {answer_text} context: {context}"
            elif qg_format == "highlight":
                source_text = f"generate question: {context[:start_pos]} <hl> {answer_text} <hl> {context[end_pos:]}"
            elif qg_format == "both":
                source_text = (
                    f"answer: {answer_text} context: {context[:start_pos]} <hl> {answer_text} <hl> {context[end_pos:]}"
                )
            else:
                raise ValueError(f"unsupported qg format: {qg_format}")

            if question_list:
                question = question_list[ind]
            else:
                question = None

            samples.append({"answer": answer_text, "source_text": source_text, "target_text": question})

    except ValueError:
        sentence_list = sentence_tokenize(normalize_text(context))

        if question_list:
            answer_list_per_sentence, question_list_per_sentence = get_answer_list_per_sentence(
                sentence_list, answer_list, question_list
            )
        else:
            answer_list_per_sentence = get_answer_list_per_sentence(sentence_list, answer_list)

        samples = []
        for sentence_ind, answer_list in enumerate(answer_list_per_sentence):
            if not answer_list:
                continue
            for answer_ind, answer in enumerate(answer_list):
                sentence = sentence_list[sentence_ind]
                sentence_list_copy = sentence_list[:]

                answer_start = answer["answer_start"]
                answer_text = answer["text"]
                answer_end = answer["answer_end"]

                sentence = f"{sentence[:answer_start]} <hl> {answer_text} <hl> {sentence[answer_end:]}"
                sentence_list_copy[sentence_ind] = sentence
                highlighted_context = " ".join(sentence_list_copy)

                if qg_format == "prepend":
                    source_text = f"answer: {answer_text} context: {context}"
                elif qg_format == "highlight":
                    source_text = f"generate question: {highlighted_context}"
                elif qg_format == "both":
                    source_text = f"answer: {answer_text} context: {highlighted_context}"
                else:
                    raise ValueError(f"unsupported qg format: {qg_format}")

                if question_list:
                    question_list = question_list_per_sentence[sentence_ind]
                    question = question_list[answer_ind]
                else:
                    question = None

                samples.append({"answer": answer_text, "source_text": source_text, "target_text": question})

    if not samples:
        samples.append({"answer": None, "source_text": None, "target_text": None})

    return samples


def postprocess_answer_extraction_output(answer_extraction_output: str):
    """
    Args:
        answer_extraction_output (str): decoded answer extraction output

    Returns:
        answer_text_list (List[str])
    """
    # parse answers
    answers = answer_extraction_output.split("<sep>")[:-1]
    # normalize and append answers
    answer_text_list = []
    for answer_text in answers:
        # append if not present
        if answer_text and (answer_text not in answer_text_list):
            answer_text_list.append(answer_text)
    return answer_text_list


def download_and_load_dataset(source_url: str, target_path: str) -> Dataset:
    attempt_download(source_url=source_url, target_path=target_path)
    data = load_json(target_path)
    return data


def remove_punctuations(text: str) -> str:
    regex = re.compile("[%s]" % re.escape(string.punctuation))
    text = regex.sub(" ", text)
    return " ".join(text.split())


def replace_multiple_spaces_with_single_whitespace(text: str) -> str:
    return re.sub("\\s+", " ", text)


def remove_citations(text: str) -> str:
    """
    Removes the citations that consist of a pair of brackets having a substring
    containing at least one digit inside them.
    Args:
        text (str):

    Returns:

    """
    text = re.sub("\[[a-zA-Z]\]", "", text)
    return re.sub(r"\[(\s|\w)*\d+(\s|\w)*\]", "", text)


def handle_ugly_case(text: str) -> str:
    pattern = r"(?<=\d.)(/)(?=\d.)"
    return re.sub(pattern, "-", text)


def normalize_text(text: str) -> str:
    text = text.strip()
    text = replace_multiple_spaces_with_single_whitespace(text)
    text = remove_citations(text)
    text = handle_ugly_case(text)
    return text
