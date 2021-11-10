import json
import os
import urllib.request
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import yaml


def safe_download(target_path: str, source_url: str, source_url2=None, min_bytes=1e0, error_msg="") -> None:
    """Attempts to download file from source_url or source_url2, checks and removes incomplete downloads < min_bytes"""
    file = Path(target_path)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        print(f"Downloading {source_url} to {file}...")
        urllib.request.urlretrieve(
            source_url,
            target_path,
        )
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        file.unlink(missing_ok=True)  # remove partial downloads
        print(f"ERROR: {e}\nRe-attempting {source_url2 or source_url} to {file}...")
        urllib.request.urlretrieve(
            source_url2 or source_url,
            target_path,
        )
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            file.unlink(missing_ok=True)  # remove partial downloads
            print(f"ERROR: {assert_msg}\n{error_msg}")
        print("")


def attempt_download(source_url: str, target_path: str) -> None:
    target_path = Path(str(target_path).strip().replace("'", ""))
    if not target_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        safe_download(target_path=str(target_path), source_url=source_url)


def load_json(load_path, object_hook=None):
    """
    Loads json formatted data (given as "data") from load_path
    Example inputs:
        load_path: "dirname/squad.json"
    """
    # read from path
    with open(load_path, encoding="utf-8") as json_file:
        data = json.load(json_file, object_hook=object_hook)
    return data


def save_json(obj: Dict, path: str, encoding: str = "utf-8", indent: int = 4):
    """
    Save dict as json file.
    """
    with open(path, "w", encoding=encoding) as jf:
        json.dump(obj, jf, indent=indent, default=str, ensure_ascii=False)


def read_yaml(yaml_path):
    """
    Reads yaml file as dict.
    """
    with open(yaml_path) as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)

    return yaml_data


def save_yaml(dict_file, yaml_path):
    """
    Saves dict as yaml file.
    """

    Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)

    with open(yaml_path, "w") as file:
        yaml.dump(dict_file, file)


def unzip(file_path: str, dest_dir: str):
    """
    Unzips compressed .zip file.
    Example inputs:
        file_path: 'data/01_alb_id.zip'
        dest_dir: 'data/'
    """

    # unzip file
    with zipfile.ZipFile(file_path) as zf:
        zf.extractall(dest_dir)


def download_from_url(from_url: str, to_path: str):

    Path(to_path).parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(to_path):
        urllib.request.urlretrieve(
            from_url,
            to_path,
        )


def save_experiment_config(model_args, data_args, training_args):
    experiment_config = {}
    experiment_config.update(asdict(model_args))
    experiment_config.update(asdict(data_args))
    experiment_config.update(asdict(training_args))
    yaml_path = Path(training_args.output_dir) / "experiment_config.yaml"
    save_yaml(experiment_config, yaml_path)


def download_from_gdrive(url: str, save_dir: str) -> str:
    """
    Downloads file from gdrive, shows progress.
    Example inputs:
        url: 'https://drive.google.com/uc?id=10hHFuavHCofDczGSzsH1xPHgTgAocOl1'
        save_dir: 'data/'
    """
    import gdown

    # create save_dir if not present
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    # download file
    filepath = gdown.download(url, save_dir, quiet=False)
    return filepath


def download_from_gdrive_and_unzip(url: str, save_dir: str) -> str:
    save_dir = save_dir + os.path.sep
    # download zip file
    filepath = download_from_gdrive(url, save_dir)
    # extract zip file
    unzip(filepath, str(Path(filepath).parent))
    # remove zip file
    os.remove(filepath)
