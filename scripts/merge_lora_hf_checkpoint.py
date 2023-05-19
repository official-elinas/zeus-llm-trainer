import argparse

import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer

parser = argparse.ArgumentParser(description="Llama model LoRA conversion")
parser.add_argument("--base-model", type=str, required=True, help="Base model path")
parser.add_argument("--lora-adapter", type=str, required=True, help="Lora adapter path")
parser.add_argument("--save-path", type=str, required=True, help="Path to save the model checkpoint")
args = parser.parse_args()

tokenizer = LlamaTokenizer.from_pretrained(args.base_model)

base_model = LlamaForCausalLM.from_pretrained(
    args.base_model,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

lora_model = PeftModel.from_pretrained(
    base_model,
    args.lora_adapter,
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)

lora_weight = lora_model.base_model.model.model.layers[
    0
].self_attn.q_proj.weight

assert torch.allclose(first_weight_old, first_weight)

# merge weights - new merging method from peft
lora_model = lora_model.merge_and_unload()

lora_model.train(False)

assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

LlamaForCausalLM.save_pretrained(
    base_model, args.save_path, state_dict=deloreanized_sd, max_shard_size="11000MB"
)
