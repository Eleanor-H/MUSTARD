from dataclasses import dataclass


@dataclass
class llama_model:
    model: str = "LlamaForCausalLM"
    tokenizer: str = "LlamaTokenizer"