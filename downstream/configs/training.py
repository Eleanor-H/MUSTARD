# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from dataclasses import dataclass
from typing import ClassVar

import os

WORK_DIR = os.environ.get('WORK_DIR')

@dataclass
class train_config:
    model_name: str=os.path.join(WORK_DIR, "model_hub/llama2-hf/Llama-2-7b-hf")
    enable_fsdp: bool= False 
    run_validation: bool=True
    batch_size_training: int=8
    num_epochs: int=10
    num_workers_dataloader: int=4
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=8
    dataset = "mustard_dataset"
    micro_batch_size: int=8
    peft_method: str = "lora" # lora, None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = os.path.join(WORK_DIR, "checkpoint/llama2_gsm8k_finetune")
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str=os.path.join(WORK_DIR, "checkpoint/llama2_gsm8k_finetune_fsdp") # will be used if using FSDP
    dist_checkpoint_folder: str="finetuned" # will be used if using FSDP
    save_optimizer: bool=True # will be used if using FSDP

    
    
    