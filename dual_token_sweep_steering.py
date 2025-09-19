import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import sys
import argparse
import copy

class DualTokenSweepSteering:
    def __init__(self, model_path, sae_model_path_1, sae_model_path_2, vec_file_path_1, vec_file_path_2,
                 target_layer_name_1, target_layer_name_2, layer_names, device_id, torch_dtype_str, 
                 tokenizer_path, model_revision, trust_remote_code_flag, cache_dir_path, use_fast_tokenizer,
                 padding_side, model_max_length, steering_strength_1, steering_strength_2,
                 normalization_method_1, normalization_method_2, normalization_eps_1, normalization_eps_2,
                 intervention_type_1, intervention_type_2, gradient_checkpointing,
                 debug_mode, sweep_output_dir, baseline_output_file, generation_max_length,
                 temperature_val, top_k_val, top_p_val, repetition_penalty_val, do_sample_flag,
                 num_beams, early_stopping_flag, pad_token_id_val, eos_token_id_val, seed_val,
                 max_sweep_iterations, min_position_gap, sweep_mode, position_selection_strategy,
                 overlap_handling_method):
        
        if not model_path:
            raise ValueError("model_path is required")
        if not sae_model_path_1:
            raise ValueError("sae_model_path_1 is required")
        if not sae_model_path_2:
            raise ValueError("sae_model_path_2 is required")
        if not vec_file_path_1:
            raise ValueError("vec_file_path_1 is required")
        if not vec_file_path_2:
            raise ValueError("vec_file_path_2 is required")
        if not target_layer_name_1:
            raise ValueError("target_layer_name_1 is required")
        if not target_layer_name_2:
            raise ValueError("target_layer_name_2 is required")
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
        if not model_max_length:
            raise ValueError("model_max_length is required")
        if steering_strength_1 is None:
            raise ValueError("steering_strength_1 is required")
        if steering_strength_2 is None:
            raise ValueError("steering_strength_2 is required")
        if not normalization_method_1:
            raise ValueError("normalization_method_1 is required")
        if not normalization_method_2:
            raise ValueError("normalization_method_2 is required")
        if normalization_eps_1 is None:
            raise ValueError("normalization_eps_1 is required")
        if normalization_eps_2 is None:
            raise ValueError("normalization_eps_2 is required")
        if not intervention_type_1:
            raise ValueError("intervention_type_1 is required")
        if not intervention_type_2:
            raise ValueError("intervention_type_2 is required")
        if gradient_checkpointing is None:
            raise ValueError("gradient_checkpointing is required")
        if debug_mode is None:
            raise ValueError("debug_mode is required")
        if not sweep_output_dir:
            raise ValueError("sweep_output_dir is required")
        if not baseline_output_file:
            raise ValueError("baseline_output_file is required")
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
        if not max_sweep_iterations:
            raise ValueError("max_sweep_iterations is required")
        if not min_position_gap:
            raise ValueError("min_position_gap is required")
        if not sweep_mode:
            raise ValueError("sweep_mode is required")
        if not position_selection_strategy:
            raise ValueError("position_selection_strategy is required")
        if not overlap_handling_method:
            raise ValueError("overlap_handling_method is required")
            
        self.model_path = model_path
        self.sae_model_path_1 = sae_model_path_1
        self.sae_model_path_2 = sae_model_path_2
        self.vec_file_path_1 = vec_file_path_1
        self.vec_file_path_2 = vec_file_path_2
        self.target_layer_name_1 = target_layer_name_1
        self.target_layer_name_2 = target_layer_name_2
        self.layer_names = layer_names
        self.device_id = device_id
        self.torch_dtype_str = torch_dtype_str
        self.tokenizer_path = tokenizer_path
        self.model_revision = model_revision
        self.trust_remote_code_flag = trust_remote_code_flag
        self.cache_dir_path = cache_dir_path
        self.use_fast_tokenizer = use_fast_tokenizer
        self.padding_side = padding_side
        self.model_max_length = model_max_length
        self.steering_strength_1 = steering_strength_1
        self.steering_strength_2 = steering_strength_2
        self.normalization_method_1 = normalization_method_1
        self.normalization_method_2 = normalization_method_2
        self.normalization_eps_1 = normalization_eps_1
        self.normalization_eps_2 = normalization_eps_2
        self.intervention_type_1 = intervention_type_1
        self.intervention_type_2 = intervention_type_2
        self.gradient_checkpointing = gradient_checkpointing
        self.debug_mode = debug_mode
        self.sweep_output_dir = sweep_output_dir
        self.baseline_output_file = baseline_output_file
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
        self.max_sweep_iterations = max_sweep_iterations
        self.min_position_gap = min_position_gap
        self.sweep_mode = sweep_mode
        self.position_selection_strategy = position_selection_strategy
        self.overlap_handling_method = overlap_handling_method
        
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
            
        self.current_token_position_1 = 0
        self.current_token_position_2 = 0
        self.steering_vector_1 = None
        self.steering_vector_2 = None
        self.hook_handles = []
        self.sweep_results = []
        self.baseline_length = 0
        
    def load_components(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            revision=self.model_revision,
            trust_remote_code=self.trust_remote_code_flag,
            cache_dir=self.cache_dir_path,
            use_fast=self.use_fast_tokenizer,
            model_max_length=self.model_max_length,
            padding_side=self.padding_side,
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
        
        sae_state_1 = torch.load(self.sae_model_path_1, map_location=self.device)
        sae_state_2 = torch.load(self.sae_model_path_2, map_location=self.device)
        
        input_dim_1 = None
        hidden_dim_1 = None
        input_dim_2 = None
        hidden_dim_2 = None
        
        for key, value in sae_state_1.items():
            if "encoder.weight" in key:
                input_dim_1 = value.shape[1]
                hidden_dim_1 = value.shape[0]
                break
                
        for key, value in sae_state_2.items():
            if "encoder.weight" in key:
                input_dim_2 = value.shape[1]
                hidden_dim_2 = value.shape[0]
                break
                
        if input_dim_1 is None or hidden_dim_1 is None:
            raise ValueError("Could not determine SAE 1 dimensions")
        if input_dim_2 is None or hidden_dim_2 is None:
            raise ValueError("Could not determine SAE 2 dimensions")
            
        from train_sae import SparseAutoEncoder
        
        self.sae_model_1 = SparseAutoEncoder(
            input_dim=input_dim_1,
            hidden_dim=hidden_dim_1,
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
        
        self.sae_model_2 = SparseAutoEncoder(
            input_dim=input_dim_2,
            hidden_dim=hidden_dim_2,
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
        
        self.sae_model_1.load_state_dict(sae_state_1)
        self.sae_model_1.eval()
        self.sae_model_2.load_state_dict(sae_state_2)
        self.sae_model_2.eval()
        
        with open(self.vec_file_path_1, 'r') as f:
            vec_data_1 = json.load(f)
            
        with open(self.vec_file_path_2, 'r') as f:
            vec_data_2 = json.load(f)
            
        self.feature_indices_1 = []
        self.feature_values_1 = []
        self.feature_indices_2 = []
        self.feature_values_2 = []
        
        for key, value in vec_data_1.items():
            if key != "name":
                self.feature_indices_1.append(int(key))
                self.feature_values_1.append(float(value))
                
        for key, value in vec_data_2.items():
            if key != "name":
                self.feature_indices_2.append(int(key))
                self.feature_values_2.append(float(value))
                
        self.feature_indices_1 = torch.tensor(self.feature_indices_1, device=self.device)
        self.feature_values_1 = torch.tensor(self.feature_values_1, device=self.device)
        self.feature_indices_2 = torch.tensor(self.feature_indices_2, device=self.device)
        self.feature_values_2 = torch.tensor(self.feature_values_2, device=self.device)
        
    def dual_steering_hook(self, module, input_tensor, output_tensor):
        module_name = None
        for name, mod in self.model.named_modules():
            if mod is module:
                module_name = name
                break
                
        if isinstance(output_tensor, tuple):
            hidden_states = output_tensor[0]
        else:
            hidden_states = output_tensor
            
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        if self.debug_mode:
            print(f"Applying dual steering at positions {self.current_token_position_1}, {self.current_token_position_2}")
            print(f"Hidden states shape: {hidden_states.shape}")
            print(f"Module name: {module_name}")
            
        if (module_name == self.target_layer_name_1 and 
            self.current_token_position_1 < seq_len and 
            self.current_token_position_1 >= 0):
            
            target_activations_1 = hidden_states[:, self.current_token_position_1, :]
            
            with torch.no_grad():
                encoded_1 = self.sae_model_1.encoder(target_activations_1)
                sae_activations_1 = self.sae_model_1.activation(encoded_1)
                
            steering_activations_1 = sae_activations_1.clone()
            
            for i, (feature_idx, feature_val) in enumerate(zip(self.feature_indices_1, self.feature_values_1)):
                steering_activations_1[:, feature_idx] += feature_val * self.steering_strength_1
                
            if self.normalization_method_1 == "l2":
                steering_activations_1 = F.normalize(steering_activations_1, p=2, dim=-1)
            elif self.normalization_method_1 == "layer_norm":
                steering_activations_1 = F.layer_norm(steering_activations_1, steering_activations_1.shape[-1:], eps=self.normalization_eps_1)
            elif self.normalization_method_1 == "batch_norm":
                steering_activations_1 = F.batch_norm(steering_activations_1.unsqueeze(0), None, None, eps=self.normalization_eps_1).squeeze(0)
            elif self.normalization_method_1 == "none":
                pass
            else:
                raise ValueError("Invalid normalization_method_1")
                
            with torch.no_grad():
                steered_hidden_1 = self.sae_model_1.decoder(steering_activations_1)
                
            if self.intervention_type_1 == "add":
                hidden_states[:, self.current_token_position_1, :] = target_activations_1 + steered_hidden_1
            elif self.intervention_type_1 == "replace":
                hidden_states[:, self.current_token_position_1, :] = steered_hidden_1
            elif self.intervention_type_1 == "interpolate":
                alpha = self.steering_strength_1
                hidden_states[:, self.current_token_position_1, :] = (1 - alpha) * target_activations_1 + alpha * steered_hidden_1
            else:
                raise ValueError("Invalid intervention_type_1")
                
        if (module_name == self.target_layer_name_2 and 
            self.current_token_position_2 < seq_len and 
            self.current_token_position_2 >= 0):
            
            target_activations_2 = hidden_states[:, self.current_token_position_2, :]
            
            with torch.no_grad():
                encoded_2 = self.sae_model_2.encoder(target_activations_2)
                sae_activations_2 = self.sae_model_2.activation(encoded_2)
                
            steering_activations_2 = sae_activations_2.clone()
            
            for i, (feature_idx, feature_val) in enumerate(zip(self.feature_indices_2, self.feature_values_2)):
                steering_activations_2[:, feature_idx] += feature_val * self.steering_strength_2
                
            if self.normalization_method_2 == "l2":
                steering_activations_2 = F.normalize(steering_activations_2, p=2, dim=-1)
            elif self.normalization_method_2 == "layer_norm":
                steering_activations_2 = F.layer_norm(steering_activations_2, steering_activations_2.shape[-1:], eps=self.normalization_eps_2)
            elif self.normalization_method_2 == "batch_norm":
                steering_activations_2 = F.batch_norm(steering_activations_2.unsqueeze(0), None, None, eps=self.normalization_eps_2).squeeze(0)
            elif self.normalization_method_2 == "none":
                pass
            else:
                raise ValueError("Invalid normalization_method_2")
                
            with torch.no_grad():
                steered_hidden_2 = self.sae_model_2.decoder(steering_activations_2)
                
            if self.intervention_type_2 == "add":
                hidden_states[:, self.current_token_position_2, :] = target_activations_2 + steered_hidden_2
            elif self.intervention_type_2 == "replace":
                hidden_states[:, self.current_token_position_2, :] = steered_hidden_2
            elif self.intervention_type_2 == "interpolate":
                alpha = self.steering_strength_2
                hidden_states[:, self.current_token_position_2, :] = (1 - alpha) * target_activations_2 + alpha * steered_hidden_2
            else:
                raise ValueError("Invalid intervention_type_2")
                
        if isinstance(output_tensor, tuple):
            return (hidden_states,) + output_tensor[1:]
        else:
            return hidden_states
        
    def register_hooks(self):
        target_modules = []
        
        for layer_name in self.layer_names:
            if layer_name == self.target_layer_name_1 or layer_name == self.target_layer_name_2:
                for name, module in self.model.named_modules():
                    if name == layer_name:
                        target_modules.append(module)
                        break
                        
        if not target_modules:
            for name, module in self.model.named_modules():
                if self.target_layer_name_1 in name or self.target_layer_name_2 in name:
                    target_modules.append(module)
                    
        if not target_modules:
            raise ValueError(f"Target layers {self.target_layer_name_1}, {self.target_layer_name_2} not found")
            
        for target_module in target_modules:
            hook_handle = target_module.register_forward_hook(self.dual_steering_hook)
            self.hook_handles.append(hook_handle)
        
    def remove_hooks(self):
        for hook_handle in self.hook_handles:
            hook_handle.remove()
        self.hook_handles = []
        
    def generate_baseline(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
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
                eos_token_id=self.eos_token_id_val
            )
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.baseline_length = outputs[0].shape[0] - inputs.input_ids.shape[1]
        
        with open(self.baseline_output_file, 'w') as f:
            f.write(generated_text)
            
        return generated_text, self.baseline_length
        
    def generate_with_dual_steering(self, prompt, pos1, pos2):
        self.current_token_position_1 = pos1
        self.current_token_position_2 = pos2
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        self.register_hooks()
        
        try:
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
                    eos_token_id=self.eos_token_id_val
                )
                
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        finally:
            self.remove_hooks()
            
        return generated_text
        
    def perform_position_sweep(self, prompt):
        os.makedirs(self.sweep_output_dir, exist_ok=True)
        
        baseline_text, baseline_len = self.generate_baseline(prompt)
        
        if baseline_len < 2:
            raise ValueError("Baseline generation too short for position sweep")
            
        sweep_log = []
        iteration_count = 0
        
        for pos1 in range(1, baseline_len):
            if iteration_count >= self.max_sweep_iterations:
                break
                
            for pos2 in range(pos1 + self.min_position_gap, baseline_len):
                if iteration_count >= self.max_sweep_iterations:
                    break
                    
                if self.debug_mode:
                    print(f"Sweeping positions: token1={pos1}, token2={pos2}")
                    
                try:
                    steered_text = self.generate_with_dual_steering(prompt, pos1, pos2)
                    
                    sweep_result = {
                        'iteration': iteration_count,
                        'position_1': pos1,
                        'position_2': pos2,
                        'generated_text': steered_text,
                        'baseline_length': baseline_len,
                        'sweep_mode': self.sweep_mode,
                        'position_gap': pos2 - pos1
                    }
                    
                    sweep_log.append(sweep_result)
                    self.sweep_results.append(sweep_result)
                    
                    output_file = os.path.join(self.sweep_output_dir, f"sweep_{iteration_count:04d}_pos{pos1}-{pos2}.txt")
                    with open(output_file, 'w') as f:
                        f.write(steered_text)
                        
                    iteration_count += 1
                    
                except Exception as e:
                    if self.debug_mode:
                        print(f"Error at positions {pos1}, {pos2}: {e}")
                    continue
                    
        sweep_summary_file = os.path.join(self.sweep_output_dir, "sweep_summary.json")
        with open(sweep_summary_file, 'w') as f:
            json.dump({
                'sweep_metadata': {
                    'total_iterations': iteration_count,
                    'baseline_length': baseline_len,
                    'max_position_1': baseline_len - 1,
                    'max_position_2': baseline_len - 1,
                    'min_position_gap': self.min_position_gap,
                    'sweep_mode': self.sweep_mode,
                    'position_selection_strategy': self.position_selection_strategy,
                    'overlap_handling_method': self.overlap_handling_method
                },
                'sweep_results': sweep_log
            }, f, indent=2)
            
        return sweep_log

def main():
    parser = argparse.ArgumentParser(description='Dual-token position sweep steering with SAE-based interventions')
    parser.add_argument('--model_path', required=True, help='Path to the model')
    parser.add_argument('--sae_model_path_1', required=True, help='Path to first SAE model')
    parser.add_argument('--sae_model_path_2', required=True, help='Path to second SAE model')
    parser.add_argument('--vec_file_path_1', required=True, help='First vector file path')
    parser.add_argument('--vec_file_path_2', required=True, help='Second vector file path')
    parser.add_argument('--target_layer_name_1', required=True, help='First target layer name')
    parser.add_argument('--target_layer_name_2', required=True, help='Second target layer name')
    parser.add_argument('--layer_names', required=True, help='JSON string of layer names')
    parser.add_argument('--device_id', type=int, required=True, help='Device ID')
    parser.add_argument('--torch_dtype_str', required=True, help='Torch dtype string')
    parser.add_argument('--tokenizer_path', required=True, help='Tokenizer path')
    parser.add_argument('--model_revision', required=True, help='Model revision')
    parser.add_argument('--trust_remote_code_flag', required=True, help='Trust remote code flag')
    parser.add_argument('--cache_dir_path', required=True, help='Cache directory path')
    parser.add_argument('--use_fast_tokenizer', required=True, help='Use fast tokenizer')
    parser.add_argument('--padding_side', required=True, help='Padding side')
    parser.add_argument('--model_max_length', type=int, required=True, help='Model max length')
    parser.add_argument('--steering_strength_1', type=float, required=True, help='First steering strength')
    parser.add_argument('--steering_strength_2', type=float, required=True, help='Second steering strength')
    parser.add_argument('--normalization_method_1', required=True, help='First normalization method')
    parser.add_argument('--normalization_method_2', required=True, help='Second normalization method')
    parser.add_argument('--normalization_eps_1', type=float, required=True, help='First normalization epsilon')
    parser.add_argument('--normalization_eps_2', type=float, required=True, help='Second normalization epsilon')
    parser.add_argument('--intervention_type_1', required=True, help='First intervention type')
    parser.add_argument('--intervention_type_2', required=True, help='Second intervention type')
    parser.add_argument('--gradient_checkpointing', required=True, help='Gradient checkpointing')
    parser.add_argument('--debug_mode', required=True, help='Debug mode')
    parser.add_argument('--sweep_output_dir', required=True, help='Sweep output directory')
    parser.add_argument('--baseline_output_file', required=True, help='Baseline output file')
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
    parser.add_argument('--max_sweep_iterations', type=int, required=True, help='Maximum sweep iterations')
    parser.add_argument('--min_position_gap', type=int, required=True, help='Minimum position gap')
    parser.add_argument('--sweep_mode', required=True, help='Sweep mode')
    parser.add_argument('--position_selection_strategy', required=True, help='Position selection strategy')
    parser.add_argument('--overlap_handling_method', required=True, help='Overlap handling method')
    
    args = parser.parse_args()
    
    layer_names = json.loads(args.layer_names)
    
    prompt = input("Enter prompt: ")
    
    steerer = DualTokenSweepSteering(
        args.model_path, args.sae_model_path_1, args.sae_model_path_2, args.vec_file_path_1, args.vec_file_path_2,
        args.target_layer_name_1, args.target_layer_name_2, layer_names, args.device_id, args.torch_dtype_str,
        args.tokenizer_path, args.model_revision, args.trust_remote_code_flag == "True", args.cache_dir_path, args.use_fast_tokenizer == "True",
        args.padding_side, args.model_max_length, args.steering_strength_1, args.steering_strength_2,
        args.normalization_method_1, args.normalization_method_2, args.normalization_eps_1, args.normalization_eps_2,
        args.intervention_type_1, args.intervention_type_2, args.gradient_checkpointing == "True",
        args.debug_mode == "True", args.sweep_output_dir, args.baseline_output_file, args.generation_max_length,
        args.temperature_val, args.top_k_val, args.top_p_val, args.repetition_penalty_val, args.do_sample_flag == "True",
        args.num_beams, args.early_stopping_flag == "True", args.pad_token_id_val, args.eos_token_id_val, args.seed_val,
        args.max_sweep_iterations, args.min_position_gap, args.sweep_mode, args.position_selection_strategy,
        args.overlap_handling_method
    )
    
    steerer.load_components()
    sweep_results = steerer.perform_position_sweep(prompt)
    
    print(f"Position sweep completed. {len(sweep_results)} iterations performed.")
    print(f"Results saved to {args.sweep_output_dir}")

if __name__ == "__main__":
    main()