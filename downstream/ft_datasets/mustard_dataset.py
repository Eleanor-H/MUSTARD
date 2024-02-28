import os
import copy
import json
from torch.utils.data import Dataset
import torch
import random

WORK_DIR = os.environ['WORK_DIR']

def get_mustard_dataset(dataset_config, tokenizer, split):
    dataset = MustardDataset(dataset_config, tokenizer, split)
    return dataset


class MustardDataset(Dataset):
    def __init__(self, data_args, tokenizer, prompt_type='qa', split='train'):
        super().__init__()
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.split = split
        self.prompt_type = prompt_type

        self.prompt = (
        f"You are an expert in math. Answer the following math word problem:\nQ:{{question}}\nA:"
        )

        self.data = self.load_data(name=data_args.data_tag, path=data_args.path, split=split)

        print('Data size:', len(self.data))
        print('Data format:', self.data[0])
        print('Max length:', max([len(d['input_ids']) for d in self.data]))

    def apply_prompt_template(self, sample):
        return {
            "text": self.prompt.format(
                question=sample['informal_statement'],
                answer=sample['informal_proof'],
            )
        }
        
    def load_data(self, path=f'{WORK_DIR}/data/mustard/batch_230912_sampled', name='subset_filtered', split='train') -> list:

        dataset = list()
        if split == 'train':
            for root, dirs, files in os.walk(os.path.join(path, name)):
                for file in files:
                    full_name = os.path.join(root, file)
                    with open(full_name, 'r', encoding='utf8') as f:
                        content = json.load(f)
                        prompt = self.apply_prompt_template(content)["text"]
                        example = prompt + content['informal_proof']
                        # example = prompt + content['formal_proof']
                        prompt = torch.tensor(
                            self.tokenizer.encode(prompt), dtype=torch.int64
                        )
                        example = self.tokenizer.encode(example)
                        example.append(self.tokenizer.eos_token_id)
                        example = torch.tensor(
                            example, dtype=torch.int64
                        )
                        padding = self.data_args.data_max_length - example.shape[0]
                        if padding > 0:
                            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
                        elif padding < 0:
                            example = example[: self.data_args.data_max_length]
                        labels = copy.deepcopy(example)
                        labels[: len(prompt)] = -1
                        example_mask = example.ge(0)
                        label_mask = labels.ge(0)
                        example[~example_mask] = 0
                        labels[~label_mask] = 0
                        example_mask = example_mask.float()
                        label_mask = label_mask.float()

                        dataset.append({
                            "input_ids": example,
                            "labels": labels,
                            "attention_mask":example_mask,
                        })
            return dataset
        elif split == 'test' or split == 'eval':
            for root, dirs, files in os.walk(os.path.join(path, 'total')):
                for file in files:
                    full_name = os.path.join(root, file)
                    with open(full_name, 'r', encoding='utf8') as f:
                        content = json.load(f)
                        prompt = self.apply_prompt_template(content)["text"]
                        example = prompt + content['informal_proof']
                        # example = prompt + content['formal_proof']
                        prompt = torch.tensor(
                            self.tokenizer.encode(prompt), dtype=torch.int64
                        )
                        example = self.tokenizer.encode(example)
                        example.append(self.tokenizer.eos_token_id)
                        example = torch.tensor(
                            example, dtype=torch.int64
                        )
                        padding = self.data_args.data_max_length - example.shape[0]
                        if padding > 0:
                            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
                        elif padding < 0:
                            example = example[: self.data_args.data_max_length]
                        labels = copy.deepcopy(example)
                        labels[: len(prompt)] = -1
                        example_mask = example.ge(0)
                        label_mask = labels.ge(0)
                        example[~example_mask] = 0
                        labels[~label_mask] = 0
                        example_mask = example_mask.float()
                        label_mask = label_mask.float()

                        dataset.append({
                            "input_ids": example,
                            "labels": labels,
                            "attention_mask":example_mask,
                        })
            return random.choice(dataset, 200)
        else:
            raise NotImplementedError
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx]['input_ids'],
            'labels': self.data[idx]['labels'],
            'attention_mask': self.data[idx]['attention_mask'],
        }

