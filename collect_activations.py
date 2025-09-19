import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import sys
import argparse

class ActivationCollector:
    def __init__(self, model_path, layer_names, target_layer_name, device_id, 
                 batch_size_limit, sequence_length_max, output_file_path, tokenizer_path,
                 model_revision, trust_remote_code_flag, torch_dtype_str, cache_dir_path,
                 use_fast_tokenizer, padding_side, truncation_side, model_max_length):
        
        if not model_path:
            raise ValueError("model_path is required")
        if not layer_names:
            raise ValueError("layer_names is required") 
        if not target_layer_name:
            raise ValueError("target_layer_name is required")
        if device_id is None:
            raise ValueError("device_id is required")
        if not batch_size_limit:
            raise ValueError("batch_size_limit is required")
        if not sequence_length_max:
            raise ValueError("sequence_length_max is required")
        if not output_file_path:
            raise ValueError("output_file_path is required")
        if not tokenizer_path:
            raise ValueError("tokenizer_path is required")
        if not model_revision:
            raise ValueError("model_revision is required")
        if trust_remote_code_flag is None:
            raise ValueError("trust_remote_code_flag is required")
        if not torch_dtype_str:
            raise ValueError("torch_dtype_str is required")
        if not cache_dir_path:
            raise ValueError("cache_dir_path is required")
        if use_fast_tokenizer is None:
            raise ValueError("use_fast_tokenizer is required")
        if not padding_side:
            raise ValueError("padding_side is required")
        if not truncation_side:
            raise ValueError("truncation_side is required")
        if not model_max_length:
            raise ValueError("model_max_length is required")
            
        self.model_path = model_path
        self.layer_names = layer_names
        self.target_layer_name = target_layer_name
        self.device_id = device_id
        self.batch_size_limit = batch_size_limit
        self.sequence_length_max = sequence_length_max
        self.output_file_path = output_file_path
        self.tokenizer_path = tokenizer_path
        self.model_revision = model_revision
        self.trust_remote_code_flag = trust_remote_code_flag
        self.torch_dtype_str = torch_dtype_str
        self.cache_dir_path = cache_dir_path
        self.use_fast_tokenizer = use_fast_tokenizer
        self.padding_side = padding_side
        self.truncation_side = truncation_side
        self.model_max_length = model_max_length
        
        self.device = torch.device(f"cuda:{device_id}")
        
        if torch_dtype_str == "float16":
            self.torch_dtype = torch.float16
        elif torch_dtype_str == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif torch_dtype_str == "float32":
            self.torch_dtype = torch.float32
        else:
            raise ValueError("Invalid torch_dtype_str")
            
        self.activations = []
        self.hooks = []
        
    def load_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            revision=self.model_revision,
            trust_remote_code=self.trust_remote_code_flag,
            cache_dir=self.cache_dir_path,
            use_fast=self.use_fast_tokenizer,
            model_max_length=self.model_max_length,
            padding_side=self.padding_side,
            truncation_side=self.truncation_side
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            revision=self.model_revision,
            trust_remote_code=self.trust_remote_code_flag,
            torch_dtype=self.torch_dtype,
            cache_dir=self.cache_dir_path,
            device_map={"": self.device_id}
        )
        
        self.model.eval()
        
    def hook_function(self, module, input_tensor, output_tensor):
        if isinstance(output_tensor, tuple):
            activation_data = output_tensor[0].detach().cpu()
        else:
            activation_data = output_tensor.detach().cpu()
        self.activations.append(activation_data)
        
    def register_hooks(self):
        target_module = None
        
        for layer_name in self.layer_names:
            if layer_name == self.target_layer_name:
                for name, module in self.model.named_modules():
                    if name == layer_name:
                        target_module = module
                        break
                        
        if target_module is None:
            for name, module in self.model.named_modules():
                if self.target_layer_name in name:
                    target_module = module
                    break
                    
        if target_module is None:
            raise ValueError(f"Target layer {self.target_layer_name} not found")
            
        hook_handle = target_module.register_forward_hook(self.hook_function)
        self.hooks.append(hook_handle)
        
    def remove_hooks(self):
        for hook_handle in self.hooks:
            hook_handle.remove()
        self.hooks = []
        
    def collect_activations(self, input_texts):
        self.register_hooks()
        
        all_activations = []
        
        for i in range(0, len(input_texts), self.batch_size_limit):
            batch_texts = input_texts[i:i+self.batch_size_limit]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.sequence_length_max
            ).to(self.device)
            
            self.activations = []
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            for activation_tensor in self.activations:
                batch_size, seq_len, hidden_dim = activation_tensor.shape
                for batch_idx in range(batch_size):
                    for seq_idx in range(seq_len):
                        activation_vector = activation_tensor[batch_idx, seq_idx, :]
                        all_activations.append(activation_vector)
                        
        self.remove_hooks()
        
        activation_tensor = torch.stack(all_activations)
        torch.save(activation_tensor, self.output_file_path)
        
def main():
    parser = argparse.ArgumentParser(description='Collect activations for SAE training')
    parser.add_argument('--model_path', required=True, help='Path to the model')
    parser.add_argument('--layer_names', required=True, help='JSON string of layer names')
    parser.add_argument('--target_layer_name', required=True, help='Target layer name')
    parser.add_argument('--device_id', type=int, required=True, help='Device ID')
    parser.add_argument('--batch_size_limit', type=int, required=True, help='Batch size limit')
    parser.add_argument('--sequence_length_max', type=int, required=True, help='Max sequence length')
    parser.add_argument('--output_file_path', required=True, help='Output file path')
    parser.add_argument('--tokenizer_path', required=True, help='Tokenizer path')
    parser.add_argument('--model_revision', required=True, help='Model revision')
    parser.add_argument('--trust_remote_code_flag', required=True, help='Trust remote code flag')
    parser.add_argument('--torch_dtype_str', required=True, help='Torch dtype string')
    parser.add_argument('--cache_dir_path', required=True, help='Cache directory path')
    parser.add_argument('--use_fast_tokenizer', required=True, help='Use fast tokenizer')
    parser.add_argument('--padding_side', required=True, help='Padding side')
    parser.add_argument('--truncation_side', required=True, help='Truncation side')
    parser.add_argument('--model_max_length', type=int, required=True, help='Model max length')
    
    args = parser.parse_args()
    
    model_path = args.model_path
    layer_names = json.loads(args.layer_names)
    target_layer_name = args.target_layer_name
    device_id = args.device_id
    batch_size_limit = args.batch_size_limit
    sequence_length_max = args.sequence_length_max
    output_file_path = args.output_file_path
    tokenizer_path = args.tokenizer_path
    model_revision = args.model_revision
    trust_remote_code_flag = args.trust_remote_code_flag == "True"
    torch_dtype_str = args.torch_dtype_str
    cache_dir_path = args.cache_dir_path
    use_fast_tokenizer = args.use_fast_tokenizer == "True"
    padding_side = args.padding_side
    truncation_side = args.truncation_side
    model_max_length = args.model_max_length
    
    input_texts_file = input("Enter input texts file path: ")
    if not os.path.exists(input_texts_file):
        raise ValueError("Input texts file not found")
        
    with open(input_texts_file, 'r') as f:
        input_texts = [line.strip() for line in f.readlines()]
        
    collector = ActivationCollector(
        model_path, layer_names, target_layer_name, device_id,
        batch_size_limit, sequence_length_max, output_file_path, tokenizer_path,
        model_revision, trust_remote_code_flag, torch_dtype_str, cache_dir_path,
        use_fast_tokenizer, padding_side, truncation_side, model_max_length
    )
    
    collector.load_model_and_tokenizer()
    collector.collect_activations(input_texts)

if __name__ == "__main__":
    main()