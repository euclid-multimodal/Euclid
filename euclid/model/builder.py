#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from safetensors import safe_open
from euclid.model import *
from euclid.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

class ModelAttributeError(Exception):
    pass

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def update_vision_tower(model, model_path):
    safetensor_files = []
    for root, dirs, files in os.walk(model_path):
        for file in files:
            if file.endswith('.safetensors'):
                safetensor_files.append(os.path.join(root, file))
    
    for file in safetensor_files:
        
        # Open the safetensor file
        with safe_open(file, framework="pt", device="cpu") as f:
            # Get all tensor names
            tensor_names = f.keys()
            
            # Find all tensors containing 'vision_tower'
            vision_tower_tensors = [name for name in tensor_names if 'vision_tower' in name]
            
            for tensor_name in vision_tower_tensors:
                # Get the tensor
                tensor = f.get_tensor(tensor_name)
                current = model
                name_parts = tensor_name.split('.')
                for part in name_parts[:-1]:
                    if not hasattr(current, part):
                        raise ModelAttributeError(f'Attribute {part} not found in model')
                    current = getattr(current, part)
                tensor = tensor.to(dtype=getattr(current, name_parts[-1]).dtype, device=getattr(current, name_parts[-1]).device)
                setattr(current, name_parts[-1], torch.nn.Parameter(tensor, requires_grad=getattr(current, name_parts[-1]).requires_grad))
    
    return model

def update_everything(model, model_path):
    safetensor_files = []
    for file in os.listdir(model_path):
        if file.endswith('.safetensors'):
            safetensor_files.append(os.path.join(model_path, file))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensor files found in {model_path}")
    for file in safetensor_files:
        # Open the safetensor file
        with safe_open(file, framework="pt", device="cpu") as f:
            # Get all tensor names
            tensor_names = f.keys()
            
            # Find all tensors containing 'vision_tower'
            vision_tower_tensors = [name for name in tensor_names]
            
            for tensor_name in vision_tower_tensors:
                # Get the new tensor
                new_tensor = f.get_tensor(tensor_name)
                
                # Navigate to the correct attribute in the model
                current = model
                name_parts = tensor_name.split('.')
                for part in name_parts[:-1]:
                    if not hasattr(current, part):
                        raise ModelAttributeError(f'Attribute {part} not found in model')
                    current = getattr(current, part)
                
                # Get the old tensor
                old_tensor = getattr(current, name_parts[-1])
                
                # Update the tensor
                new_tensor = new_tensor.to(dtype=old_tensor.dtype)
                new_tensor = new_tensor.to(device=old_tensor.device)
                new_param = torch.nn.Parameter(new_tensor, requires_grad=old_tensor.requires_grad)
                old_num_parameters = count_parameters(model)
                setattr(current, name_parts[-1], new_param)
                new_num_parameters = count_parameters(model)
                # if old_num_parameters != new_num_parameters:
                #     print(f"Updated tensor: {tensor_name}")
                #     print(f"Old number of parameters: {old_num_parameters}")
                #     print(f"New number of parameters: {new_num_parameters}")
                
        print(f'Model updated with tensors from {file}')
    return model

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'euclid' in model_name.lower():
        # Load Euclid model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/Euclid#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from euclid.model.language_model.euclid_llama import EuclidConfig
            lora_cfg_pretrained = EuclidConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading Euclid from base model...')
            model = EuclidLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional Euclid weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading Euclid from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = EuclidMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            elif 'qwen' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = EuclidQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = EuclidLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = EuclidMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = EuclidMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            elif 'qwen' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = EuclidQwen2ForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = EuclidLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'euclid' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if model.config.tune_vision_tower or model.config.only_tune_vision_tower:
        model = update_vision_tower(model, model_path)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
