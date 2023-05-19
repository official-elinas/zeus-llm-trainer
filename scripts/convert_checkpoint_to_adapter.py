import argparse
import torch
from transformers import LlamaForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, set_peft_model_state_dict

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default="elinas/llama-13b-hf-transformers-4.29")
parser.add_argument('--checkpoint_path', default="/home/elinas/llama-13b/checkpoint-600/pytorch_model.bin")
parser.add_argument('--save_path', default="./lora-adapter")
parser.add_argument('--lora_rank', type=int, default=64)
parser.add_argument('--lora_alpha', type=int, default=128)
parser.add_argument('--lora_dropout', type=float, default=0.05)
parser.add_argument('--target_modules', nargs='+', default=['q_proj','k_proj','v_proj','o_proj'])
args = parser.parse_args()

model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map='auto',
)
print('Loaded original model')

model = prepare_model_for_int8_training(model)
print('Prepared model for 8bit')

# LoRA config must match yours!
config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

model = get_peft_model(model, config)
print('Get peft model using config')

full_checkpoint = torch.load(args.checkpoint_path)
print('Load pytorch_model.bin checkpoint')

set_peft_model_state_dict(model, full_checkpoint)
print('Set peft model state dict')

model.save_pretrained(args.save_path)
print('Saved model')
