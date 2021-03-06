import logging
import os
from typing import Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BertForQuestionAnswering, BertTokenizerFast
from transformers.hf_argparser import DataClass

from utils.file import download_from_gdrive_and_unzip
from utils.torch import assert_not_all_frozen, freeze_embeds

PRETRAINED_NAME_TO_GDRIVE_URL = {
    "mt5-small-3task-highlight-tquad2": "https://drive.google.com/uc?id=1Cnovcib1I276GmJVOGa33jySIwOthIa7",
    "mt5-small-3task-prepend-tquad2": "https://drive.google.com/uc?id=1JG14mynmu-b3Dy2UDJr4AyJQyuW-uabh",
    "mt5-base-3task-highlight-tquad2": "https://drive.google.com/uc?id=1hVhR5hQHcIVKj5pPgvYkcl1WWDDHpOFL",
    "mt5-small-3task-highlight-combined3": "https://drive.google.com/uc?id=10hHFuavHCofDczGSzsH1xPHgTgAocOl1",
    "mt5-base-3task-highlight-combined3": "https://drive.google.com/uc?id=1W8PXgx6VDaThDdLNkL-HVWb1MNcQdxwp",
}

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)


class MT5Model:
    def __init__(
        self,
        model_name_or_path: str = "mt5-small-3task-highlight-combined3",
        tokenizer_name_or_path: str = None,
        freeze_embeddings: bool = False,
        cache_dir: Optional[str] = None,
        use_cuda: bool = None,
    ):
        # try downloading pretrained files from gdrive
        if model_name_or_path in PRETRAINED_NAME_TO_GDRIVE_URL.keys():
            download_dir = "data/pretrained/" + model_name_or_path + "/"
            weight_path = "data/pretrained/" + model_name_or_path + "/" + "/pytorch_model.bin"
            if not os.path.isfile(weight_path):
                download_from_gdrive_and_unzip(PRETRAINED_NAME_TO_GDRIVE_URL[model_name_or_path], download_dir)
                logger.info(f"pretrained model is downloaded to {download_dir}")
            else:
                logger.info(f"using pretrained model at {download_dir}")
            model_name_or_path = download_dir

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
        )
        if tokenizer_name_or_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=cache_dir)
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
            except:
                tokenizer = AutoTokenizer.from_pretrained(
                    os.path.join(model_name_or_path, "tokenizer_config.json"), cache_dir=cache_dir
                )
        assert model.__class__.__name__ in ["MT5ForConditionalGeneration"]
        self.model = model
        self.tokenizer = tokenizer
        self.type = "mt5"

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

        if freeze_embeddings:
            logger.info("freezing embeddings of the model")
            freeze_embeds(self.model)
            assert_not_all_frozen(self.model)

        self.model.resize_token_embeddings(len(self.tokenizer))


class BertModel:
    def __init__(
        self,
        model_name_or_path: str = "dbmdz/bert-base-turkish-cased",
        tokenizer_name_or_path: str = None,
        freeze_embeddings: bool = False,
        cache_dir: Optional[str] = None,
        use_cuda: bool = None,
    ):

        model = BertForQuestionAnswering.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
        )
        if tokenizer_name_or_path is not None:
            tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name_or_path, cache_dir=cache_dir)
        else:
            try:
                tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path, cache_dir=cache_dir)
            except:
                tokenizer = BertTokenizerFast.from_pretrained(
                    os.path.join(model_name_or_path, "tokenizer_config.json"), cache_dir=cache_dir
                )
        assert model.__class__.__name__ in ["BertForQuestionAnswering"]
        self.model = model
        self.tokenizer: BertTokenizerFast = tokenizer
        self.type = "bert"

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

        if freeze_embeddings:
            logger.info("freezing embeddings of the model")
            freeze_embeds(self.model)
            assert_not_all_frozen(self.model)

        self.model.resize_token_embeddings(len(self.tokenizer))
