"""
This
"""

import os
import json
import re
import random
import builtins
from pathlib import Path
from logging import getLogger
import logging
from contextlib import contextmanager
from shutil import rmtree, copytree
import multiprocessing as mp

import jsonlines
from tqdm import tqdm

import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
from formal_system_client import LeanClient

logger = getLogger()
BUCKETS = list('ABCDEFGHIJK')

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError
        return self.get(attr, None)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def build_files_slide_window(data_path, tokenized_data_path, num_pieces, full_tokenizer, min_length):
    with open(data_path, 'r', encoding='utf8') as f:
        print('reading lines')
        lines = f.readlines()
        lines = [line.replace('\n', ' <|endoftext|> ') for line in lines]  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
    all_len = len(lines)
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    for i in tqdm(range(num_pieces)):
        sublines = lines[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]
        if i == num_pieces - 1:
            sublines.extend(lines[all_len // num_pieces * (i + 1):])  # 把尾部例子添加到最后一个piece
        sublines = [full_tokenizer.tokenize(line) for line in sublines if
                    len(line) > min_length]  # 只考虑长度超过min_length的句子
        sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]
        full_line = []
        for subline in sublines:
            full_line.append(full_tokenizer.convert_tokens_to_ids('<|endoftext|>'))  # 文章开头添加MASK表示文章开始
            full_line.extend(subline)
            full_line.append(full_tokenizer.convert_tokens_to_ids('<|endoftext|>'))  # 文章之间添加CLS表示文章结束
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'w') as f:
            for id in full_line:
                f.write(str(id) + ' ')
    logger.info('finish')

def split_file(data_paths, split_data_path, num_pieces):
    all_lines = []
    for data_path in data_paths:
        with open(data_path, 'r', encoding='utf8') as f:
            print('reading lines')
            all_lines.extend(f.readlines())
    random.shuffle(all_lines)
    lines = all_lines
    all_len = len(lines)
    if not os.path.exists(split_data_path):
        os.makedirs(split_data_path, exist_ok=True)
    for i in tqdm(range(num_pieces)):
        sublines = lines[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]
        if i == num_pieces - 1:
            sublines.extend(lines[all_len // num_pieces * (i + 1):])  # 把尾部例子添加到最后一个piece
        with open(split_data_path + 'split_train_{}.txt'.format(i), 'w') as f:
            for subline in sublines:
                f.write(subline)
    logger.info('finish')

def cal_on_track_info(on_track_rates):
    if len(on_track_rates) == 0:
        return torch.tensor([0,0,0.0])
    on_track_rates = torch.tensor(on_track_rates)
    total_correct_num = (on_track_rates == 1).sum().item()
    total_num = (on_track_rates != -1).sum().item()
    total_sum = ((on_track_rates != -1).int() * on_track_rates).sum().item()
    not_total_correct_sum = ((on_track_rates != -1).int() * (on_track_rates != 1).int() * on_track_rates).sum().item()

    return torch.tensor([total_correct_num, total_num, total_sum, not_total_correct_sum])

def init_distributed(port=40101, rank_and_world_size=(None, None)):

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    os.environ['MASTER_ADDR'] = 'localhost'

    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ['SLURM_NTASKS'])
            rank = int(os.environ['SLURM_PROCID'])
            os.environ['MASTER_ADDR'] = os.environ['HOSTNAME']
        except Exception:
            logger.info('distributed training not available')
            world_size, rank = 1, 0
            return world_size, rank

    try:
        os.environ['MASTER_PORT'] = str(port)
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank)
    except Exception:
        world_size, rank = 1, 0
        logger.info('distributed training not available')

    return world_size, rank

def append_to_csv(dataframe, path):
    """
    Append dataframe to csv, create new csv file if the path is not alreadly existed
    """
    if not os.path.isfile(path):
        pd.DataFrame(dataframe).to_csv(path, header='column_names', index=None)
    else: # else it exists so append without writing the header
        pd.DataFrame(dataframe).to_csv(path, mode='a', header=False, index=None)

def append_to_jsonl(data_dict, path):
    with jsonlines.open(path, mode='a') as writer:
        for data in data_dict:
            writer.write(json.dumps(data))

def setup_log(filename, name):
    logger = logging.getLogger(name)   # > set up a new name for a new logger
    logger.propagate = False
    logger.setLevel(logging.DEBUG)  # here is the missing line

    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_handler = logging.FileHandler(filename, mode='w')
    log_handler.setLevel(logging.DEBUG)
    log_handler.setFormatter(log_format)

    logger.addHandler(log_handler)

    return logger

@contextmanager
def memmap(*args, **kwargs):
    pointer = np.memmap(*args, **kwargs)
    yield pointer
    del pointer

def reset_folder_(p):
    path = Path(p)
    rmtree(path, ignore_errors = True)
    path.mkdir(exist_ok = True, parents = True)

def copy_folder_(src, dest):
    src = Path(src)
    dest = Path(dest)
    copytree(src, dest, dirs_exist_ok=True)

def get_bucket(proofsize):
    # -- get bucket
    bucket = None
    # -- inifinte proofsize
    if proofsize > 1000:
        bucket = BUCKETS[0]
        return bucket
    if proofsize > 20:
        bucket = BUCKETS[1]
    else:
        # linearly projecting proofsize under 20 in 9 buckets: 20 / 9 = 2.222...
        bucket = BUCKETS[int(10 - proofsize // 2.222)]
    return bucket


def print(*objs, **kwargs):
    my_prefix = f'{mp.current_process().name} :: '
    builtins.print(my_prefix, *objs, **kwargs)


def get_thm_list(filepath):
    thm_names = []
    pattern = "theorem (\S+)"
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            match = re.search(pattern, line)
            if match:
                thm_name = match.group(1)
                thm_names.append(thm_name)
            else:
                if "theorem" in line:
                    print("Not working: {}".format(line))
    return thm_names


def lean2names(thm_names, names_path):
    with open(names_path, 'w') as f:
        f.write(" \n".join(thm_names) + " ")


def extract_names(src_dir, output_dir):
    ds = os.listdir(src_dir)
    for d in ds:
        dp = os.path.join(src_dir, d)
        if os.path.isdir(dp):
            lean_files = [ f for f in os.listdir(dp) if f.endswith(".lean")]
            if len(lean_files) > 0:
                names_path = os.path.join(output_dir, "{}.names".format(d))
                thm_names = []
                for lean_file in lean_files:
                    lean_path = os.path.join(dp, lean_file)
                    tn = get_thm_list(lean_path)
                    thm_names += tn
                lean2names(thm_names, names_path)


def extract_minif2f_goals():
    test_names_path = '/home/ma-user/work/zhengying/open_domain_knowledge_representation_learning_and_reasoning/whm/hwatp/datasets/cleaned_training_data/miniF2F/test.names'
    valid_names_path = '/home/ma-user/work/zhengying/open_domain_knowledge_representation_learning_and_reasoning/whm/hwatp/datasets/cleaned_training_data/miniF2F/valid.names'

    lean_server = LeanClient()

    names = []
    with open(test_names_path, 'r') as f:
        names.extend(f.readlines())
    with open(valid_names_path, 'r') as f:
        names.extend(f.readlines())
    
    outputs = {}

    for name in tqdm(names):
        name = name.strip()
        ts = lean_server.init_search(name)
        ts = lean_server.run_tac(ts['search_id'], 0, "try {intros}")
        outputs[name] = ts['tactic_state']

    with open('/home/ma-user/work/zhengying/open_domain_knowledge_representation_learning_and_reasoning/whm/hwatp/datasets/cleaned_training_data/miniF2F/test_valid.index', 'w') as f:
        json.dump(outputs, f)


if __name__ == '__main__':
    # extract_minif2f_goals()
    # src_dir = "/home/ma-user/work/zhengying/open_domain_knowledge_representation_learning_and_reasoning/whm/hwatp/lean_gym/_target/deps/INT/src/"
    # output_dir = "/home/ma-user/work/zhengying/open_domain_knowledge_representation_learning_and_reasoning/whm/hwatp/datasets/cleaned_training_data/INT/"
    # extract_names(src_dir, output_dir)

    # pl2hf(pl_checkpoint='/cache/epoch=00-step=2033-validation_loss=1.00.ckpt',
    #       hf_mode_type='/cache/gpt2-L28-pact-1epoch-ft-7epoch/',
    #       output_dir='/cache/epoch=00-step=2033-validation_loss=1.00/')

    pass