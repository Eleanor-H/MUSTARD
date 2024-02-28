# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import sys
import warnings
from typing import List

import fire
import torch
from model_utils import load_model, load_peft_model
from peft import PeftConfig, PeftModel
from transformers import (GPT2Tokenizer, LlamaConfig, LlamaForCausalLM,
                          LlamaTokenizer, default_data_collator)

sys.path.append('../')
sys.path.append(os.environ['PWD'])
import re

import numpy as np

from configs.datasets import gsm8k_dataset
from utils.dataset_utils import get_preprocessed_dataset


def main(
    model_name,
    peft_model: str=None,
    prompt_type: str='few_shot_cot',
    quantization: bool=False,
    max_new_tokens: int= 256, #The maximum numbers of tokens to generate
    min_new_tokens: int=0, #The minimum numbers of tokens to generate
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=0.95, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=0.8, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    **kwargs
):
    tokenizer = None
    if 'llama' in model_name.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
    elif 'gpt2' in model_name.lower():
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens(            
            {
                "pad_token": "<|endoftext|>",
            }
        )
    else:
        raise NotImplementedError

    dataset_config = gsm8k_dataset(prompt_type=prompt_type)
    dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()
    # chats = format_tokens(dialogs, tokenizer)
    total = 0
    correct_list = [] 

    with torch.no_grad():
        for idx, data in enumerate(dataset_test):
            print('*************************')
            print("{}st data".format(idx+1))
            

            tokens= data['input_ids'].unsqueeze(0).to("cuda")
            outputs = model.generate(
                input_ids=tokens,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs
            )

            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            pred = answer_cleansing(output_text, prompt_type)
            gt = data['ground_truth']
            
            print(output_text)
            print("pred", pred)
            print("gt", gt)
            print('*************************')

            # Checking answer ...
            correct = (np.array([pred]) == np.array([gt])).sum().item()
            correct_list.append(correct)
            total += 1
            
        
    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    print("accuracy : {}".format(accuracy))
            

def answer_cleansing(pred, prompt_type):
    if prompt_type in ('zero_shot', 'few_shot'):
        answer_trigger = "The answer is:"
        pred = pred.split(answer_trigger)[-1]
    elif prompt_type in ('zero_shot_cot', 'few_shot_cot'):
        answer_trigger = "Let's think step by step."
        pred = pred.split(answer_trigger)[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if prompt_type in ('zero_shot', 'few_shot'):
            pred = pred[0]
        elif prompt_type in ('zero_shot_cot', 'few_shot_cot'):
            pred = pred[-1]

    return pred


if __name__ == "__main__":
    fire.Fire(main)
