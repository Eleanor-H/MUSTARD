import datasets
from .utils import Concatenator
from transformers import LlamaTokenizer
from torch.utils.data import Dataset
import torch
import copy
import os

PROMPT_DICT = { 
    "zero_shot": (f"""You are an expert in math. Answer the following math word problem with an arabic numeral.
Question: {{question}}
Answer: The answer is: """),
    "zero_shot_cot": (f"""You are an expert in math. Answer the following math word problem. Show your thought and answer an arabic numeral.
Question: {{question}}
Answer: Let's think step by step."""),
    "few_shot": (f"""You are an expert in math. Answer the following math word problem with an arabic numeral.
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: The answer is 39. 

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: The answer is 33. 

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: The answer is 8.

Q: {{question}}
A: """),
    "few_shot_cot": (f"""You are an expert in math. Answer the following math word problem.
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39. 

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33. 

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

Q: {{question}}
A: Let's think step by step."""),
}


class GSM8kDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split, data_max_length=1024):
        self.split = split
        # self.data = datasets.load_dataset("gsm8k", 'main', split=split)
        self.data = datasets.load_from_disk(os.path.join(dataset_config.path, split))
        self.tokenizer = tokenizer
        self.max_words = data_max_length
        self.prompt_type = dataset_config.prompt_type

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        example = self.data[index]
        prompt = PROMPT_DICT[self.prompt_type].format(question = example['question'])
        ground_truth = example['answer'].split('#### ')[-1]

        if self.split in ['train', 'eval']:
            example = prompt + example['answer'].replace('#### ', 'The answer is ')

            prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
            example = self.tokenizer.encode(example)
            example.append(self.tokenizer.eos_token_id)
            example = torch.tensor(
                example, dtype=torch.int64
            )
            padding = self.max_words - example.shape[0]
            if padding > 0:
                example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                example = example[: self.max_words]
            labels = copy.deepcopy(example)
            labels[: len(prompt)] = -1
            example_mask = example.ge(0)
            label_mask = labels.ge(0)
            example[~example_mask] = 0
            labels[~label_mask] = 0
            example_mask = example_mask.float()
            label_mask = label_mask.float()

            return {
                "input_ids": example,
                "labels": labels,
                "attention_mask":example_mask,
                # "ground_truth": ground_truth,
            }
        
        else:
            example = prompt
            example = self.tokenizer.encode(example)
            example = torch.tensor(
                example, dtype=torch.int64
            )

            return {
                "input_ids": example,
                "ground_truth": ground_truth,
            }
