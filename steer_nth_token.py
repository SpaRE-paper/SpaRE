import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import sys
import argparse

class NthTokenSAESteering:
    def __init__(self, model_path, sae_model_path, vec_file_path, target_layer_name,
                 layer_names, device_id, torch_dtype_str, tokenizer_path,
                 model_revision, trust_remote_code_flag, cache_dir_path, use_fast_tokenizer,
                 padding_side, truncation_side, model_max_length, target_token_position,
                 steering_strength, normalization_method, normalization_eps, 
                 intervention_type, gradient_checkpointing, memory_efficient_mode,
                 debug_mode, output_file_path):
        
        if not model_path:
            raise ValueError("model_path is required")
        if not sae_model_path:
            raise ValueError("sae_model_path is required")
        if not vec_file_path:
            raise ValueError("vec_file_path is required")
        if not target_layer_name:
            raise ValueError("target_layer_name is required")
        if not layer_names:
            raise ValueError("layer_names is required")
        if device_id is None:
            raise ValueError("device_id is required")
        if not torch_dtype_str:
            raise ValueError("torch_dtype_str is required")
        if not tokenizer_path:
            raise ValueError("tokenizer_path is required")
        if not model_revision:
            raise ValueError("model_revision is required")
        if trust_remote_code_flag is None:
            raise ValueError("trust_remote_code_flag is required")
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
        if target_token_position is None:
            raise ValueError("target_token_position is required")
        if steering_strength is None:
            raise ValueError("steering_strength is required")
        if not normalization_method:
            raise ValueError("normalization_method is required")
        if normalization_eps is None:
            raise ValueError("normalization_eps is required")
        if not intervention_type:
            raise ValueError("intervention_type is required")
        if gradient_checkpointing is None:
            raise ValueError("gradient_checkpointing is required")
        if memory_efficient_mode is None:
            raise ValueError("memory_efficient_mode is required")
        if debug_mode is None:
            raise ValueError("debug_mode is required")
        if not output_file_path:
            raise ValueError("output_file_path is required")
            
        self.model_path = model_path
        self.sae_model_path = sae_model_path
        self.vec_file_path = vec_file_path
        self.target_layer_name = target_layer_name
        self.layer_names = layer_names
        self.device_id = device_id
        self.torch_dtype_str = torch_dtype_str
        self.tokenizer_path = tokenizer_path
        self.model_revision = model_revision
        self.trust_remote_code_flag = trust_remote_code_flag
        self.cache_dir_path = cache_dir_path
        self.use_fast_tokenizer = use_fast_tokenizer
        self.padding_side = padding_side
        self.truncation_side = truncation_side
        self.model_max_length = model_max_length
        self.target_token_position = target_token_position
        self.steering_strength = steering_strength
        self.normalization_method = normalization_method
        self.normalization_eps = normalization_eps
        self.intervention_type = intervention_type
        self.gradient_checkpointing = gradient_checkpointing
        self.memory_efficient_mode = memory_efficient_mode
        self.debug_mode = debug_mode
        self.output_file_path = output_file_path
        
        self.device = torch.device(f"cuda:{device_id}")
        
        if torch_dtype_str == "float16":
            self.torch_dtype = torch.float16
        elif torch_dtype_str == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif torch_dtype_str == "float32":
            self.torch_dtype = torch.float32
        else:
            raise ValueError("Invalid torch_dtype_str")
            
        self.current_token_position = 0
        self.steering_vector = None
        self.hook_handle = None
        
    def load_components(self):
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
        
        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        self.model.eval()
        
        sae_state = torch.load(self.sae_model_path, map_location=self.device)
        
        input_dim = None
        hidden_dim = None
        
        for key, value in sae_state.items():
            if "encoder.weight" in key:
                input_dim = value.shape[1]
                hidden_dim = value.shape[0]
                break
                
        if input_dim is None or hidden_dim is None:
            raise ValueError("Could not determine SAE dimensions")
            
        from train_sae import SparseAutoEncoder
        
        self.sae_model = SparseAutoEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            sparsity_coeff=0.0,
            learning_rate_encoder=0.001,
            learning_rate_decoder=0.001,
            weight_decay_val=0.0,
            eps_val=1e-5,
            normalization_type="layer",
            momentum_val=0.1,
            affine_transform=True,
            track_running_stats=True,
            bias_encoder=True,
            bias_decoder=True,
            init_method="xavier_uniform",
            init_scale=1.0
        ).to(self.device)
        
        self.sae_model.load_state_dict(sae_state)
        self.sae_model.eval()
        
        with open(self.vec_file_path, 'r') as f:
            vec_data = json.load(f)
            
        self.feature_indices = []
        self.feature_values = []
        
        for key, value in vec_data.items():
            if key != "name":
                self.feature_indices.append(int(key))
                self.feature_values.append(float(value))
                
        self.feature_indices = torch.tensor(self.feature_indices, device=self.device)
        self.feature_values = torch.tensor(self.feature_values, device=self.device)
        
    def steering_hook(self, module, input_tensor, output_tensor):
        if self.current_token_position == self.target_token_position:
            if isinstance(output_tensor, tuple):
                hidden_states = output_tensor[0]
            else:
                hidden_states = output_tensor
                
            batch_size, seq_len, hidden_dim = hidden_states.shape
            
            if self.debug_mode:
                print(f"Applying steering at position {self.current_token_position}")
                print(f"Hidden states shape: {hidden_states.shape}")
                
            target_activations = hidden_states[:, self.target_token_position, :]
            
            with torch.no_grad():
                encoded = self.sae_model.encoder(target_activations)
                sae_activations = self.sae_model.activation(encoded)
                
            steering_activations = sae_activations.clone()
            
            for i, (feature_idx, feature_val) in enumerate(zip(self.feature_indices, self.feature_values)):
                steering_activations[:, feature_idx] += feature_val * self.steering_strength
                
            if self.normalization_method == "l2":
                steering_activations = F.normalize(steering_activations, p=2, dim=-1)
            elif self.normalization_method == "layer_norm":
                steering_activations = F.layer_norm(steering_activations, steering_activations.shape[-1:], eps=self.normalization_eps)
            elif self.normalization_method == "batch_norm":
                steering_activations = F.batch_norm(steering_activations.unsqueeze(0), None, None, eps=self.normalization_eps).squeeze(0)
            elif self.normalization_method == "none":
                pass
            else:
                raise ValueError("Invalid normalization_method")
                
            with torch.no_grad():
                steered_hidden = self.sae_model.decoder(steering_activations)
                
            if self.intervention_type == "add":
                hidden_states[:, self.target_token_position, :] = target_activations + steered_hidden
            elif self.intervention_type == "replace":
                hidden_states[:, self.target_token_position, :] = steered_hidden
            elif self.intervention_type == "interpolate":
                alpha = self.steering_strength
                hidden_states[:, self.target_token_position, :] = (1 - alpha) * target_activations + alpha * steered_hidden
            else:
                raise ValueError("Invalid intervention_type")
                
            if self.debug_mode:
                print(f"Applied {self.intervention_type} intervention with strength {self.steering_strength}")
                
            if isinstance(output_tensor, tuple):
                return (hidden_states,) + output_tensor[1:]
            else:
                return hidden_states
                
        return output_tensor
        
    def register_hook(self):
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
            
        self.hook_handle = target_module.register_forward_hook(self.steering_hook)
        
    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            
    def generate_with_steering(self, prompt, generation_params):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        self.register_hook()
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    **generation_params
                )
                
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        finally:
            self.remove_hook()
            
        return generated_text

def main():
    parser = argparse.ArgumentParser(description='Apply SAE-based steering to specific token positions')
    parser.add_argument('--model_path', required=True, help='Path to the model')
    parser.add_argument('--sae_model_path', required=True, help='Path to SAE model')
    parser.add_argument('--vec_file_path', required=True, help='Vector file path')
    parser.add_argument('--target_layer_name', required=True, help='Target layer name')
    parser.add_argument('--layer_names', required=True, help='JSON string of layer names')
    parser.add_argument('--device_id', type=int, required=True, help='Device ID')
    parser.add_argument('--torch_dtype_str', required=True, help='Torch dtype string')
    parser.add_argument('--tokenizer_path', required=True, help='Tokenizer path')
    parser.add_argument('--model_revision', required=True, help='Model revision')
    parser.add_argument('--trust_remote_code_flag', required=True, help='Trust remote code flag')
    parser.add_argument('--cache_dir_path', required=True, help='Cache directory path')
    parser.add_argument('--use_fast_tokenizer', required=True, help='Use fast tokenizer')
    parser.add_argument('--padding_side', required=True, help='Padding side')
    parser.add_argument('--truncation_side', required=True, help='Truncation side')
    parser.add_argument('--model_max_length', type=int, required=True, help='Model max length')
    parser.add_argument('--target_token_position', type=int, required=True, help='Target token position')
    parser.add_argument('--steering_strength', type=float, required=True, help='Steering strength')
    parser.add_argument('--normalization_method', required=True, help='Normalization method')
    parser.add_argument('--normalization_eps', type=float, required=True, help='Normalization epsilon')
    parser.add_argument('--intervention_type', required=True, help='Intervention type')
    parser.add_argument('--gradient_checkpointing', required=True, help='Gradient checkpointing')
    parser.add_argument('--memory_efficient_mode', required=True, help='Memory efficient mode')
    parser.add_argument('--debug_mode', required=True, help='Debug mode')
    parser.add_argument('--output_file_path', required=True, help='Output file path')
    
    args = parser.parse_args()
    
    layer_names = json.loads(args.layer_names)
    
    prompt = input("Enter prompt: ")
    
    generation_params_str = input("Enter generation parameters (JSON): ")
    generation_params = json.loads(generation_params_str)
    
    steerer = NthTokenSAESteering(
        args.model_path, args.sae_model_path, args.vec_file_path, args.target_layer_name, layer_names,
        args.device_id, args.torch_dtype_str, args.tokenizer_path, args.model_revision,
        args.trust_remote_code_flag == "True", args.cache_dir_path, args.use_fast_tokenizer == "True", args.padding_side,
        args.truncation_side, args.model_max_length, args.target_token_position, args.steering_strength,
        args.normalization_method, args.normalization_eps, args.intervention_type, args.gradient_checkpointing == "True",
        args.memory_efficient_mode == "True", args.debug_mode == "True",
        args.output_file_path
    )
    
    steerer.load_components()
    result = steerer.generate_with_steering(prompt, generation_params)
    
    with open(args.output_file_path, 'w') as f:
        f.write(result)
        
    print(f"Generated text saved to {args.output_file_path}")

if __name__ == "__main__":
    main()