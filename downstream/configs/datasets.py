# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
import os

WORK_DIR = os.environ['WORK_DIR']

@dataclass
class mustard_dataset:
    dataset: str = "mustard_dataset"
    path: str = f"{WORK_DIR}/data/mustard/batch_230912_sampled"
    train_split: str = "train"
    test_split: str = "eval"
    refresh: bool = False
    data_tag: str = 'subset_filtered'
    train_on_inputs: bool = False
    data_max_length: int = 1024


@dataclass 
class gsm8k_dataset:
    dataset: str = "gsm8k_dataset"
    path: str = f"{WORK_DIR}/data/gsm8k"
    train_split: str  = 'train'
    eval_split: str = 'eval'
    test_split: str = 'test'
    data_tag: str = 'main'
    prompt_type: str = 'zero_shot'
    data_max_length: int = 512