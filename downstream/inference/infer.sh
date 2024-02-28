# source $WORK_DIR/env/mustard/bin/activate

export LLAMA2="$WORK_DIR/model_hub/llama2-hf/Llama-2-7b-hf"
export PEFT_LLAMA="$WORK_DIR/checkpoint/llama2_gsm8k_finetune"
export GPT2="$WORK_DIR/model_hub/gpt2-large"
export PEFT_GPT2="$WORK_DIR/checkpoint/gpt2_gsm8k_finetune"

CUDA_VISIBLE_DEVICES=0 python inference/infer_mwp.py --model_name $GPT2 --max_new_tokens 128 --temperature=0.9 --top_p=0.95  --top_k=50

python inference/infer_atp/formal_system_server.py --lean_gym_dir $WORK_DIR/package/lean-gym &CUDA_VISIBLE_DEVICES=0 python inference/infer_atp/eval_search.py