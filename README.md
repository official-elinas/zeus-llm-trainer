# Zeus LLM Trainer

## Table of Contents
* [Local Setup](#local-setup)  
* [LoRA Training](#lora-training) 
* [Finetuning](#finetuning)
* [Roadmap](#roadmap)
* [Development Log](#development-log)

## Currently Supported Features
* Full [LoRA](#lora-training) and [finetune](#finetuning) support - see run examples below on run configurations for each.
* All HuggingFace Models supported out of the box, though some might have special configuration requirements and may not support all listed features.
* Saving of full LoRA model and adapters - Context: there was a change in the PEFT library which stopped saving the adapter every checkpoint.
* Simple prompt templating engine for LLMs and your own can be made easily. Examples can be found in the `./templates` [directory here](https://github.com/official-elinas/zeus-llm-trainer/tree/main/templates).
  Call your unique template with `--prompt_template_name='template_name_str'` or it will default to Alpaca format.
* One time tokenization
   * The trainer will automatically tokenize your dataset once and reload in subsequent from the `./tokenized` directory.
* Numerous arguments including, but not limited to (use `python finetune.py -h` for a full list of arguments or take a look at `finetune.py`)
   * bf16 and fp16 support (`--use_bf16` OR `--use_fp16`)
   * Optimizer parameter such as `adamw_torch`, `adamw_bnb_8bit`, and more with the default `adamw_torch_fused`.
     Use `--optim='optimizer_name'` to change this. 
   * Validation dataset split (`--val_set_size=num_samples`)
      * Will split your dataset into `train` and `val` datasets with the number of samples defined by `--val_set_size=num_samples` 
        or set it to 0 not perform validation. 
   * Warmup Ratio (`--warmup_ratio=0.06` is the default)
   * Logging Steps (`--logging_steps=5` is the default)
   * Seed for training consistency between runs (`--seed=42` is the default)
* **[LLaMA Only]** Optional attention replacement techniques for memory reduction and speed (must be installed)
   * `flash-attention` (`--use_flash_attn`)
   * `xformers` (`--use_xformers`)
* Optimization Techniques
   * 8bit/int8 training (LoRA and Finetune - default unless alternate precision is specified, ie. bf16, fp16, 4bit)
   * 4bit/int4 training (QLoRA `--train_4bit`)
   * DeepSpeed - disabled by default - pass in a config like (`--deepspeed='/path/to/deepspeed_config.json'`)
   * Fully Sharded Data Parallel (FSDP) - disabled by default - (example `--fsdp_params "full_shard auto_wrap"`)
   * Gradient Checkpointing (`--use_gradient_checkpointing`) - saves significant memory at the cost of quite a bit of speed.
* Wandb Logging
   * Project name (`wandb_project='project-name'`)
   * Run name (`wandb_run_name='run-name'` - default random)
   * Wandb watch (`--wandb_watch='all'` options: false | gradients | all - default False)
* Other Features
   * Alternate batch size calculation using `--global_batch_size=<global_bsz>` instead of `--gradient_accumulation_steps=<num_steps>`
   * Gradient normalization option (`max_grad_norm=1.0` default HF value)


## Local Setup
1. Install dependencies in a virtualenv preferably 
    * Create the venv - `python -m venv venv`
    * Activate the venv - `source venv/bin/actiate`
    * Install the requirements - `pip install -r requirements.txt`
   
2. If you're using `flash_attn` or `xformers` install the `ninja` package **first** and manually install either one in
   `optional_requirements.txt`

3. For `bitsandbytes` Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17). 
   If you are doing 8bit training, at the time of writing there seems to still exist a bug leading to OOM instances when 
   saving in `bitsansbytes>0.37.2` so it's recommended to install that version if you will be doing **8bit/int8 training**.

*Note that `bitsandbytes` is not officially supported on Windows, nor is serious training recommended on Windows.*

## Run Examples

### LoRA Training

Example usage:

```bash
python finetune.py \
    --base_model 'elinas/llama-7b-hf-transformers-4.29' \
    --data_path 'hf/dataset_path' \
    --output_dir './lora-alpaca'
```
`--data_path` can be a HuggingFace dataset or a local json/jsonl file but must have the **instruction, input, output** schema.

We can also tweak our hyperparameters:

```bash
python finetune.py \
    --base_model 'elinas/llama-7b-hf-transformers-4.29' \
    --data_path 'dataset.json' \
    --output_dir './lora-alpaca' \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 4 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 1024 \
    --val_set_size 2000 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length
```

Example LoRA with DDP Usage (2 GPUs, adjust top line based on GPU count):
```bash
OMP_NUM_THREADS=12 WORLD_SIZE=2 torchrun --nproc_per_node=2 --master_port=1234 finetune.py \
    --base_model='elinas/llama-7b-hf-transformers-4.29' \
    --data_path='dataset.json' \
    --num_train_epochs=3 \
    --cutoff_len=2048 \
    --group_by_length \
    --val_set_size=2000 \
    --output_dir='./7b-lora' \
    --lora_target_modules='[q_proj,v_proj,k_proj,o_proj]' \
    --lora_r=128 \
    --lora_alpha=256 \
    --gradient_accumulation_steps=4 \
    --per_device_train_batch_size=16 \
    --train_on_inputs=True \
    --seed=42
```
#### Merge LoRA Adapter into HF/PyTorch Model Format
Use `scripts/merge_lora_hf_checkpoint.py` and the arguments provided in the file to convert your `adapter_model.bin` to a full model.

You may also use the adapter directly without converting using applications like [text-generation-webui](https://github.com/oobabooga/text-generation-webui)

### Finetuning
Example Finetune with FSDP + DeepSpeed (both optional)
```bash
OMP_NUM_THREADS=12 WORLD_SIZE=2 torchrun --nproc_per_node=2 --master_port=1234 finetune.py \
    --base_model='elinas/llama-7b-hf-transformers-4.29' \
    --data_path='dataset.json' \
    --is_finetune \
    --use_bf16 \
    --num_train_epochs=3 \
    --cutoff_len=2048 \
    --group_by_length \
    --val_set_size=2000 \
    --output_dir='./7b-finetune' \
    --gradient_accumulation_steps=4 \
    --per_device_train_batch_size=16 \
    --train_on_inputs=True \
    --seed=42 \
    --fsdp_params='full_shard auto_wrap' \
    --deepspeed='path/to/deepspeed_config.json'
```
Note we used `bf16` - you must use it or `fp16`

## **Roadmap**
- [x] Use batch per device and gradient accumulation steps to calculate global steps
- [x] Save LoRA adapter correctly every checkpoint instead of the full model
- [x] Implement 4bit QLoRA
- [x] Tokenize each unique dataset once and reload that using the same name when the original json file is passed
- [x] Implement full finetuning as an option (not LoRA)
- [x] Implement `flash-attention` for llama - https://github.com/HazyResearch/flash-attention/
- [x] Working Deepspeed support
- [ ] FP8 training using accelerate (Hopper GPUs / 4000-series)
- [ ] Add more sample templates (ie. Vicuna...)
- [ ] Improve dataset loading (multiple files, different formats)
- [ ] Implement loading arguments from JSON

## Development Log
- 2023/06/10 - **Zeus LLM Trainer Release**
   - Except for possibly needing to downgrade `bitsandbytes` to `0.37.2` to prevent OOM on 8bit training, this
     release is ready and will come with a name change as stated in the past.
   - The trainer has supported the use of DeepSpeed and more info can be found at the [DeepSpeed Repo](https://github.com/microsoft/deepspeed) 
     & [DeepSpeed HF Docs](https://huggingface.co/docs/transformers/main_classes/deepspeed) 
     and can be used in the trainer by passing the argument `--deepspeed <path_to_ds_config.json>` 
   - `--fsdp_params` can be used to pass in an FSDP (Fully Shareded Data Parallel) definition like `--fsdp_params "full_shard auto_wrap"` definition directly into
     the trainer as [outlined in the documentation here.](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.fsdp) 
     This is useful when your model can't fit into a single GPU or you want to save VRAM, all at the cost of speed.
   - The documentation has been updated below and a full changelog will be posted on the releases page. Older development
     notes have been moved to the bottom of the `README`
- 2023/06/02 - **4bit QLoRA test, release, and some notes**
   - 4bit QLoRA was tested and it performed very well in both end result and performance overall. Using a rough estimate based on
     compute I used, it is about 22% faster (or more) than 8bit LoRA, with the added benefit of fitting on smaller cards. Inference
     was not impacted much if at all compared to 8bit and generates similar outputs in 4bit. Props to Tim Dettmers for releasing this
     method and getting it integrated into Transformers.
   - Pre-tokenization is almost ready, but not quite there, though I am comfortable with everything functioning here, except for the
     untested finetune feature (which should work). I just don't have the time or resources to test a finetune right now.
   - New name will be "Zeus-llm-trainer" and the repo name will be changed which will break remotes, so make sure to update your remote
     repository or re-clone it.
   - That's really it for now, and automatic handling of pre-tokenization using Arrow format will be finished soon.
- 2023/05/30 - **pretokenization WIP & repo name change**
   - **Important:** `lora_finetune.py` has been renamed to `finetune.py`
   - Adding a way to pretokenize to save time when re-running training, especially for larger datasets
   - I'm preparing to change the repo name soon, though I have not decided on a final name.
- 2023/05/28 - **working on getting flash-attention functional**
   - The current `flash-attn` library fails to install due to a torch import error and trying to install an older
     version just results in other errors. Currently, there is an open issue about this and will be looking into it more.
   - PyTorch SDP Attention was (re)-implemented but doesn't work right, so don't use it unless you want no training to happen due to abnormal loss.
     Don't know how much time if any I will commit further to this since `flash-attn` seems like the better overall option
     and you can already use `xformers` for memory savings.
   - Another issue - the latest `bitsandbytes` still has a memory issue and will often OOM when saving a checkpoint when
     training in 8 bit. I'm not sure if this is an issue in 4bit, but I'm training a 13B QLoRA with a decent chunk of memory free on a single 3090.
   - **If you do an 8 bit lora, I recommend switching to `bitsandbytes==0.37.2` or roll the dice with the latest version.**
- 2023/05/27 - **4bit QLoRA, fp16 training and more**
   - 4bit QLora training has been implemented and is usable by calling the `--train_4bit` flag. No other configuration 
     is needed on top of the current LoRA config. 
   - LoRAs can be trained in fp16 now for improved training speed at the cost of vram, use `--train_fp16`
   - Full finetuning is implemented using `--is_finetune` - The code is a bit messy and might not work 100% right, consider this completely untested.
     <br>Note that you **need** to pass `--train_fp16` or it will default to 8bit.
     If you use this argument, all LoRA arguments and operations will be bypassed in the trainer.
   - optional optimizer selection:  `--optim="optm_name"` has been added if you don't want to use the default `adamw_torch`
     `paged_adamw_8bit` was used in the QLoRA example. [Read about the different optimizers here](https://huggingface.co/docs/transformers/v4.29.1/en/perf_train_gpu_one#optimizer) <br>     If you are unsure, just keep it default or feel free to experiment, as some can add memory savings, though generally
     at the expense of speed.
   - optional gradient checkpointing: `--use_gradient_checkpointing` which can potentially save significant memory, once 
     again at the cost of speed. **This should really only be used if you can't train the model at a batch size of 1.** [Read more on Gradient Checkpointing here](https://huggingface.co/docs/transformers/v4.18.0/en/performance#gradient-checkpointing)
- 2023/05/25 - **larger models and possible issues**
    - I've been testing larger models, specifically 30/33B (and I assume this would apply to 65B as well) but the gradient
      "explosions" are not directly due to `xformers` - the attention method might make it worse, but I have experienced
      the issue without it, using grouping and without, and currently trying to find out why. It might be my dataset, so
      that is what I will be trying to change next. 
    - The next release will include pre-tokenization and full finetuning, hence the project will be renamed to "Zeus"
- 2023/05/21 - **working on multiple things at once**
    - I've updated the TODO list below in order of priority. 
    - Some changes in dev have been made like adding `--max_grad_norm`
      which can help normalize gradients when using `xformers` with no penalty speed and an improved loss.
      This defaults at `1.0` (HF default) and will change to `0.5` if using `xformers` unless a specific value
      is passed via `max_grad_norm` ie. `--max_grad_norm=0.7`
    - `--warmup_steps` as a parameter is removed and replaced with `--warmup_ratio` as it's better to calculate your warmup as a ratio. 
      By default, it is set to `0.06` which is taken from the Microsoft LoRA example. If you do not want warmup, set 
      `--warmup_ratio=0` - though this is not recommended. 
- 2023/05/18 - **metrics: group_by_length and xformers**
    - I will be posting all benchmarks/tests I perform in the project's [wiki](https://github.com/official-elinas/alpaca-lora-optimized/wiki)
- 2023/05/18 - **testing features**
    - Currently, I have been testing features, specifically `--group_by_length` and determined that
      you should essentially always use it even if your dataset is not varied in length. An experiment was done
      with a highly varied dataset and with `--group_by_length` - it took 2:30h and without it took 4hrs exactly
      while producing lower loss and VRAM.
    - `xformers` - still doing testing on this with a 30B model. The strange loss jump could be due to 
      "exploding gradients" and I am looking into a possible solution such as tweaking the `max_grad_norm`
      parameter in the HF trainer from the default of `1.0` to a lower number, or letting the user decide (likely the latter).
      In addition, I am doing testing with and without `xformers` to get a baseline of what performance improvement can 
      be gained as well as potential memory savings and will provide an update once that testing is finished.
    - `torch.compile()` was re-implemented as the speedup can be considerable for training, although I did not see
      benchmarks for 8-bit training (or if it's supported at all)? For now, it is only compatible on Linux and if you are on Windows it will be ignored.
- 2023/05/16 - **xformers info**
    - I have tested twice with `xformers` producing strange loss that drops back down after a certain amount of steps, 
      though it might be nothing serious for a full training session. If you use it, please test with and without.
- 2023/05/14 - **Continuing rewrite**
    - Fixed LoRA adapter saving. Currently, it saves full model and adapter. 
    - Allow for usage of passing arg `--gradient_accumulation_steps=<steps>` OR `--global_batch_size=<batch_size>` 
       One must be picked over the other depending on calculation you prefer.
    - Implemented xformers as an option to replace the default attention method with `--use_xformers`
    - Argument name changes, will be documented.
  
Everything below this is "TODO" and not officially supported
------

### Inference (`generate.py`)

This file reads the foundation model from the Hugging Face model hub and the LoRA weights from `tloen/alpaca-lora-7b`, and runs a Gradio interface for inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.

Example usage:

```bash
python generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'tloen/alpaca-lora-7b'
```

### Docker Setup & Inference

1. Build the container image:

```bash
docker build -t alpaca-lora .
```

2. Run the container (you can also use `finetune.py` and all of its parameters as shown above for training):

```bash
docker run --gpus=all --shm-size 64g -p 7860:7860 -v ${HOME}/.cache:/root/.cache --rm alpaca-lora generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'tloen/alpaca-lora-7b'
```

3. Open `https://localhost:7860` in the browser

### Docker Compose Setup & Inference

1. (optional) Change desired model and weights under `environment` in the `docker-compose.yml`

2. Build and run the container

```bash
docker-compose up -d --build
```

3. Open `https://localhost:7860` in the browser

4. See logs:

```bash
docker-compose logs -f
```

5. Clean everything up:

```bash
docker-compose down --volumes --rmi all
```