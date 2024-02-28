# source $WORK_DIR/env/mustard/bin/activate

export LLAMA2_7B="$WORK_DIR/model_hub/llama2-hf/Llama-2-7b-hf"
export GPT2="$WORK_DIR/model_hub/gpt2-large"

python finetuning.py  --use_peft --peft_method lora  --model_name $LLAMA2_7B --save_model --output_dir $WORK_DIR/checkpoint/llama2_gsm8k_lora
