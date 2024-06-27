""" Manifest Utils """
import json
from argparse import Namespace
from typing import Dict
from collections import defaultdict
from tqdm.auto import tqdm

from dfcon import Directory
from dfcon.path_filter import FileFilter, DircFilter, Filter


def collect_charset(manifest_path: str, charset: Dict[str, int] = None):
    if charset is None:
        charset = defaultdict(int)
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading manifest data"):
            line = line.replace("\n", "")
            transcript = json.loads(line)["text"]
            for char in transcript:
                charset[char] += 1
    return charset


def build_charset(
    data_path: str, train_dir_name: str, valid_dir_name: str, **_
) -> Dict[str, Dict[str, int]]:
    file_filter = FileFilter().include_extention("json")
    train_filter = DircFilter().contained_path(train_dir_name)
    valid_filter = DircFilter().contained_path(valid_dir_name)

    train_filter = Filter.overlap(filters=[file_filter, train_filter])
    valid_filter = Filter.overlap(filters=[file_filter, valid_filter])

    direc = Directory(data_path).build_structure()
    train_manifests = direc.get_file_path(filters=train_filter, serialize=True)
    valid_manifests = direc.get_file_path(filters=valid_filter, serialize=True)

    train_charset = None
    for manifest_path in train_manifests:
        train_charset = collect_charset(manifest_path, train_charset)
    valid_charset = None
    for manifest_path in valid_manifests:
        valid_charset = collect_charset(manifest_path, valid_charset)

    return {"train": train_charset, "valid": valid_charset}


def get_tokens(args: Namespace):
    charset = build_charset(**vars(args))

    return set(charset["train"].keys())


def get_ds_tokens(cfg):
    return collect_charset(cfg.model.train_ds.manifest_filepath)
