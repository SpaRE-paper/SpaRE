import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import sys
import argparse

def molecule_validator(generated_string):
    # Input: generated_string (str) - molecule string
    # Output: bool - True if meets requirements, False otherwise
    # Feel free to implement your own validation logic
    raise NotImplementedError("Feel free to implement your own molecule validator")

class MoleculeSpecificActivationCollector:
    def __init__(self, model_path, layer_names, target_layer_name, device_id,
                 batch_size_limit, sequence_length_max, tokenizer_path, model_revision,
                 trust_remote_code_flag, torch_dtype_str, cache_dir_path, use_fast_tokenizer,
                 padding_side, truncation_side, model_max_length, pos_output_path, 
                 neg_output_path, generation_max_length, temperature_val, top_k_val, 
                 top_p_val, repetition_penalty_val, do_sample_flag, num_beams,
                 early_stopping_flag, pad_token_id_val, eos_token_id_val, seed_val,
                 num_generations, validation_function_path, validation_module_name,
                 validation_function_name, molecule_start_token, molecule_end_token):
        
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
        if not pos_output_path:
            raise ValueError("pos_output_path is required")
        if not neg_output_path:
            raise ValueError("neg_output_path is required")
        if not generation_max_length:
            raise ValueError("generation_max_length is required")
        if temperature_val is None:
            raise ValueError("temperature_val is required")
        if top_k_val is None:
            raise ValueError("top_k_val is required")
        if top_p_val is None:
            raise ValueError("top_p_val is required")
        if repetition_penalty_val is None:
            raise ValueError("repetition_penalty_val is required")
        if do_sample_flag is None:
            raise ValueError("do_sample_flag is required")
        if num_beams is None:
            raise ValueError("num_beams is required")
        if early_stopping_flag is None:
            raise ValueError("early_stopping_flag is required")
        if pad_token_id_val is None:
            raise ValueError("pad_token_id_val is required")
        if eos_token_id_val is None:
            raise ValueError("eos_token_id_val is required")
        if seed_val is None:
            raise ValueError("seed_val is required")
        if not num_generations:
            raise ValueError("num_generations is required")
        if not validation_function_path:
            raise ValueError("validation_function_path is required")
        if not validation_module_name:
            raise ValueError("validation_module_name is required")
        if not validation_function_name:
            raise ValueError("validation_function_name is required")
        if not molecule_start_token:
            raise ValueError("molecule_start_token is required")
        if not molecule_end_token:
            raise ValueError("molecule_end_token is required")
            
        self.model_path = model_path
        self.layer_names = layer_names
        self.target_layer_name = target_layer_name
        self.device_id = device_id
        self.batch_size_limit = batch_size_limit
        self.sequence_length_max = sequence_length_max
        self.tokenizer_path = tokenizer_path
        self.model_revision = model_revision
        self.trust_remote_code_flag = trust_remote_code_flag
        self.torch_dtype_str = torch_dtype_str
        self.cache_dir_path = cache_dir_path
        self.use_fast_tokenizer = use_fast_tokenizer
        self.padding_side = padding_side
        self.truncation_side = truncation_side
        self.model_max_length = model_max_length
        self.pos_output_path = pos_output_path
        self.neg_output_path = neg_output_path
        self.generation_max_length = generation_max_length
        self.temperature_val = temperature_val
        self.top_k_val = top_k_val
        self.top_p_val = top_p_val
        self.repetition_penalty_val = repetition_penalty_val
        self.do_sample_flag = do_sample_flag
        self.num_beams = num_beams
        self.early_stopping_flag = early_stopping_flag
        self.pad_token_id_val = pad_token_id_val
        self.eos_token_id_val = eos_token_id_val
        self.seed_val = seed_val
        self.num_generations = num_generations
        self.validation_function_path = validation_function_path
        self.validation_module_name = validation_module_name
        self.validation_function_name = validation_function_name
        self.molecule_start_token = molecule_start_token
        self.molecule_end_token = molecule_end_token
        
        torch.manual_seed(seed_val)
        
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
        
        # Load validation function
        sys.path.append(os.path.dirname(validation_function_path))
        validation_module = __import__(validation_module_name)
        self.validation_function = getattr(validation_module, validation_function_name)
        
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
        
    def extract_molecule_from_text(self, text):
        start_idx = text.find(self.molecule_start_token)
        if start_idx == -1:
            return None
            
        start_idx += len(self.molecule_start_token)
        end_idx = text.find(self.molecule_end_token, start_idx)
        
        if end_idx == -1:
            return text[start_idx:].strip()
        else:
            return text[start_idx:end_idx].strip()
    
    def generate_and_collect(self, prompts):
        self.register_hooks()
        
        positive_activations = []
        negative_activations = []
        
        for prompt in prompts:
            for generation_idx in range(self.num_generations):
                inputs = self.tokenizer(prompt, return_tensors="pt",
                                      truncation=True, max_length=self.sequence_length_max).to(self.device)
                
                self.activations = []
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_length=self.generation_max_length,
                        temperature=self.temperature_val,
                        top_k=self.top_k_val,
                        top_p=self.top_p_val,
                        repetition_penalty=self.repetition_penalty_val,
                        do_sample=self.do_sample_flag,
                        num_beams=self.num_beams,
                        early_stopping=self.early_stopping_flag,
                        pad_token_id=self.pad_token_id_val,
                        eos_token_id=self.eos_token_id_val,
                        return_dict_in_generate=True,
                        output_hidden_states=False
                    )
                    
                generated_tokens = outputs.sequences[0]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                molecule_string = self.extract_molecule_from_text(generated_text)
                
                if molecule_string is not None:
                    is_valid = self.validation_function(molecule_string)
                    
                    for i, activation_tensor in enumerate(self.activations):
                        batch_size, seq_len, hidden_dim = activation_tensor.shape
                        for batch_idx in range(batch_size):
                            for seq_idx in range(seq_len):
                                activation_vector = activation_tensor[batch_idx, seq_idx, :]
                                
                                if is_valid:
                                    positive_activations.append(activation_vector)
                                else:
                                    negative_activations.append(activation_vector)
                                    
        self.remove_hooks()
        
        if positive_activations:
            pos_tensor = torch.stack(positive_activations)
            torch.save(pos_tensor, self.pos_output_path)
        else:
            torch.save(torch.empty(0), self.pos_output_path)
            
        if negative_activations:
            neg_tensor = torch.stack(negative_activations)
            torch.save(neg_tensor, self.neg_output_path)
        else:
            torch.save(torch.empty(0), self.neg_output_path)

def main():
    parser = argparse.ArgumentParser(description='Collect molecule-specific activations for global control')
    parser.add_argument('--model_path', required=True, help='Path to the model')
    parser.add_argument('--layer_names', required=True, help='JSON string of layer names')
    parser.add_argument('--target_layer_name', required=True, help='Target layer name')
    parser.add_argument('--device_id', type=int, required=True, help='Device ID')
    parser.add_argument('--batch_size_limit', type=int, required=True, help='Batch size limit')
    parser.add_argument('--sequence_length_max', type=int, required=True, help='Max sequence length')
    parser.add_argument('--tokenizer_path', required=True, help='Tokenizer path')
    parser.add_argument('--model_revision', required=True, help='Model revision')
    parser.add_argument('--trust_remote_code_flag', required=True, help='Trust remote code flag')
    parser.add_argument('--torch_dtype_str', required=True, help='Torch dtype string')
    parser.add_argument('--cache_dir_path', required=True, help='Cache directory path')
    parser.add_argument('--use_fast_tokenizer', required=True, help='Use fast tokenizer')
    parser.add_argument('--padding_side', required=True, help='Padding side')
    parser.add_argument('--truncation_side', required=True, help='Truncation side')
    parser.add_argument('--model_max_length', type=int, required=True, help='Model max length')
    parser.add_argument('--pos_output_path', required=True, help='Positive output path')
    parser.add_argument('--neg_output_path', required=True, help='Negative output path')
    parser.add_argument('--generation_max_length', type=int, required=True, help='Generation max length')
    parser.add_argument('--temperature_val', type=float, required=True, help='Temperature value')
    parser.add_argument('--top_k_val', type=int, required=True, help='Top-k value')
    parser.add_argument('--top_p_val', type=float, required=True, help='Top-p value')
    parser.add_argument('--repetition_penalty_val', type=float, required=True, help='Repetition penalty value')
    parser.add_argument('--do_sample_flag', required=True, help='Do sample flag')
    parser.add_argument('--num_beams', type=int, required=True, help='Number of beams')
    parser.add_argument('--early_stopping_flag', required=True, help='Early stopping flag')
    parser.add_argument('--pad_token_id_val', type=int, required=True, help='Pad token ID value')
    parser.add_argument('--eos_token_id_val', type=int, required=True, help='EOS token ID value')
    parser.add_argument('--seed_val', type=int, required=True, help='Seed value')
    parser.add_argument('--num_generations', type=int, required=True, help='Number of generations')
    parser.add_argument('--validation_function_path', required=True, help='Validation function path')
    parser.add_argument('--validation_module_name', required=True, help='Validation module name')
    parser.add_argument('--validation_function_name', required=True, help='Validation function name')
    parser.add_argument('--molecule_start_token', required=True, help='Molecule start token')
    parser.add_argument('--molecule_end_token', required=True, help='Molecule end token')
    
    args = parser.parse_args()
    
    layer_names = json.loads(args.layer_names)
    
    prompts_file = input("Enter prompts file path: ")
    if not os.path.exists(prompts_file):
        raise ValueError("Prompts file not found")
        
    with open(prompts_file, 'r') as f:
        prompts = [line.strip() for line in f.readlines()]
        
    collector = MoleculeSpecificActivationCollector(
        args.model_path, layer_names, args.target_layer_name, args.device_id,
        args.batch_size_limit, args.sequence_length_max, args.tokenizer_path, args.model_revision,
        args.trust_remote_code_flag == "True", args.torch_dtype_str, args.cache_dir_path, args.use_fast_tokenizer == "True",
        args.padding_side, args.truncation_side, args.model_max_length, args.pos_output_path,
        args.neg_output_path, args.generation_max_length, args.temperature_val, args.top_k_val,
        args.top_p_val, args.repetition_penalty_val, args.do_sample_flag == "True", args.num_beams,
        args.early_stopping_flag == "True", args.pad_token_id_val, args.eos_token_id_val, args.seed_val,
        args.num_generations, args.validation_function_path, args.validation_module_name,
        args.validation_function_name, args.molecule_start_token, args.molecule_end_token
    )
    
    collector.load_model_and_tokenizer()
    collector.generate_and_collect(prompts)

if __name__ == "__main__":
    main()