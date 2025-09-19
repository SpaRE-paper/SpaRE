import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import sys
import math
import random
import copy
from collections import defaultdict
import numpy as np
import argparse

class MolecularPropertyCalculator:
    def __init__(self, property_config_file, calculation_backend_config, validation_rules_file):
        
        if not property_config_file:
            raise ValueError("property_config_file is required")
        if not calculation_backend_config:
            raise ValueError("calculation_backend_config is required")
        if not validation_rules_file:
            raise ValueError("validation_rules_file is required")
            
        raise NotImplementedError("Feel free to implement your own property calculator")
        
    def calculate_all_properties(self, smiles_string):
        # Input: smiles_string (str)
        # Output: dict with keys 'logp', 'mw', 'tpsa', 'hbd', 'hba', 'aromatic_rings', 'sa_score', 'qed'
        # Feel free to implement your own property calculation
        raise NotImplementedError("Feel free to implement your own property calculation")

class MCTSNode:
    def __init__(self, molecular_state, edit_history, parent_node, action_taken,
                 visit_threshold, expansion_depth, state_encoding_method):
        
        if not molecular_state:
            raise ValueError("molecular_state is required")
        if edit_history is None:
            raise ValueError("edit_history is required")
        if visit_threshold is None:
            raise ValueError("visit_threshold is required")
        if expansion_depth is None:
            raise ValueError("expansion_depth is required")
        if not state_encoding_method:
            raise ValueError("state_encoding_method is required")
            
        self.molecular_state = molecular_state
        self.edit_history = edit_history
        self.parent_node = parent_node
        self.action_taken = action_taken
        self.visit_threshold = visit_threshold
        self.expansion_depth = expansion_depth
        self.state_encoding_method = state_encoding_method
        
        self.children = []
        self.visit_count = 0
        self.total_reward = 0.0
        self.average_reward = 0.0
        self.is_expanded = False
        self.is_terminal = False
        self.consecutive_violations = 0
        
    def get_ucb_score(self, exploration_constant, temperature_factor, bias_term):
        if not exploration_constant:
            raise ValueError("exploration_constant is required")
        if temperature_factor is None:
            raise ValueError("temperature_factor is required")
        if bias_term is None:
            raise ValueError("bias_term is required")
            
        if self.visit_count == 0:
            return float('inf')
            
        exploitation_term = self.average_reward
        
        if self.parent_node and self.parent_node.visit_count > 0:
            exploration_term = exploration_constant * math.sqrt(
                math.log(self.parent_node.visit_count) / self.visit_count
            )
        else:
            exploration_term = exploration_constant
            
        temperature_adjustment = temperature_factor * (1.0 / (1.0 + self.visit_count))
        
        ucb_score = exploitation_term + exploration_term + temperature_adjustment + bias_term
        
        return ucb_score
        
    def select_best_child(self, exploration_constant, temperature_factor, bias_term,
                         selection_method, diversity_penalty):
        if not exploration_constant:
            raise ValueError("exploration_constant is required")
        if temperature_factor is None:
            raise ValueError("temperature_factor is required")
        if bias_term is None:
            raise ValueError("bias_term is required")
        if not selection_method:
            raise ValueError("selection_method is required")
        if diversity_penalty is None:
            raise ValueError("diversity_penalty is required")
            
        if not self.children:
            return None
            
        best_child = None
        best_score = -float('inf')
        
        for child in self.children:
            if selection_method == "ucb":
                score = child.get_ucb_score(exploration_constant, temperature_factor, bias_term)
            elif selection_method == "avg_reward":
                score = child.average_reward
            elif selection_method == "visit_count":
                score = child.visit_count
            else:
                raise ValueError("Invalid selection_method")
                
            if diversity_penalty > 0:
                similarity_penalty = 0
                for other_child in self.children:
                    if other_child != child:
                        similarity_penalty += self.calculate_state_similarity(
                            child.molecular_state, other_child.molecular_state
                        )
                score -= diversity_penalty * similarity_penalty
                
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child
        
    def calculate_state_similarity(self, state1, state2):
        # Input: state1, state2 (molecular SMILES strings)
        # Output: float similarity score (0.0 to 1.0)
        # Feel free to implement your own similarity calculation
        raise NotImplementedError("Feel free to implement your own similarity calculation")
        
    def update_statistics(self, reward, decay_factor, learning_rate):
        if reward is None:
            raise ValueError("reward is required")
        if decay_factor is None:
            raise ValueError("decay_factor is required")
        if learning_rate is None:
            raise ValueError("learning_rate is required")
            
        self.visit_count += 1
        
        old_total = self.total_reward
        self.total_reward = decay_factor * old_total + reward
        
        self.average_reward = (1 - learning_rate) * self.average_reward + learning_rate * reward
        
        if self.parent_node:
            self.parent_node.update_statistics(reward, decay_factor, learning_rate)

class RewardCalculator:
    def __init__(self, reward_config_file, constraint_definitions_file):
        
        if not reward_config_file:
            raise ValueError("reward_config_file is required")
        if not constraint_definitions_file:
            raise ValueError("constraint_definitions_file is required")
            
        raise NotImplementedError("Feel free to implement your own reward calculator")
        
        
    def calculate_total_reward(self, current_properties, previous_properties, consecutive_violations):
        # Input: current_properties (dict), previous_properties (dict or None), consecutive_violations (int)
        # Output: float reward value
        # Feel free to implement your own reward calculation
        raise NotImplementedError("Feel free to implement your own reward calculation")

class SpaREMolecularEditor:
    def __init__(self, model_path, tokenizer_path, sae_model_path, vec_file_path,
                 target_layer_name, layer_names, device_id, torch_dtype_str,
                 model_revision, trust_remote_code_flag, cache_dir_path,
                 steering_strength, normalization_method, intervention_type,
                 generation_params_json, edit_template, molecule_start_token,
                 molecule_end_token, max_edit_attempts, edit_validation_method):
        
        if not model_path:
            raise ValueError("model_path is required")
        if not tokenizer_path:
            raise ValueError("tokenizer_path is required")
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
        if not model_revision:
            raise ValueError("model_revision is required")
        if trust_remote_code_flag is None:
            raise ValueError("trust_remote_code_flag is required")
        if not cache_dir_path:
            raise ValueError("cache_dir_path is required")
        if steering_strength is None:
            raise ValueError("steering_strength is required")
        if not normalization_method:
            raise ValueError("normalization_method is required")
        if not intervention_type:
            raise ValueError("intervention_type is required")
        if not generation_params_json:
            raise ValueError("generation_params_json is required")
        if not edit_template:
            raise ValueError("edit_template is required")
        if not molecule_start_token:
            raise ValueError("molecule_start_token is required")
        if not molecule_end_token:
            raise ValueError("molecule_end_token is required")
        if max_edit_attempts is None:
            raise ValueError("max_edit_attempts is required")
        if not edit_validation_method:
            raise ValueError("edit_validation_method is required")
            
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.sae_model_path = sae_model_path
        self.vec_file_path = vec_file_path
        self.target_layer_name = target_layer_name
        self.layer_names = layer_names
        self.device_id = device_id
        self.torch_dtype_str = torch_dtype_str
        self.model_revision = model_revision
        self.trust_remote_code_flag = trust_remote_code_flag
        self.cache_dir_path = cache_dir_path
        self.steering_strength = steering_strength
        self.normalization_method = normalization_method
        self.intervention_type = intervention_type
        self.generation_params = json.loads(generation_params_json)
        self.edit_template = edit_template
        self.molecule_start_token = molecule_start_token
        self.molecule_end_token = molecule_end_token
        self.max_edit_attempts = max_edit_attempts
        self.edit_validation_method = edit_validation_method
        
        self.device = torch.device(f"cuda:{device_id}")
        
        if torch_dtype_str == "float16":
            self.torch_dtype = torch.float16
        elif torch_dtype_str == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif torch_dtype_str == "float32":
            self.torch_dtype = torch.float32
        else:
            raise ValueError("Invalid torch_dtype_str")
            
        self.hook_handle = None
        
    def load_model_components(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            revision=self.model_revision,
            trust_remote_code=self.trust_remote_code_flag,
            cache_dir=self.cache_dir_path
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
        
    def apply_edit_sequence(self, original_molecule, edit_history):
        current_molecule = original_molecule
        
        for edit_step in edit_history:
            edit_prompt = self.edit_template.format(
                molecule=current_molecule,
                edit_instruction=edit_step['instruction']
            )
            
            attempt_count = 0
            valid_edit = False
            
            while attempt_count < self.max_edit_attempts and not valid_edit:
                try:
                    inputs = self.tokenizer(edit_prompt, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs.input_ids,
                            **self.generation_params
                        )
                        
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    start_idx = generated_text.find(self.molecule_start_token)
                    if start_idx != -1:
                        start_idx += len(self.molecule_start_token)
                        end_idx = generated_text.find(self.molecule_end_token, start_idx)
                        
                        if end_idx != -1:
                            edited_molecule = generated_text[start_idx:end_idx].strip()
                        else:
                            edited_molecule = generated_text[start_idx:].strip()
                            
                        if self.validate_molecule(edited_molecule):
                            current_molecule = edited_molecule
                            valid_edit = True
                        else:
                            attempt_count += 1
                    else:
                        attempt_count += 1
                        
                except Exception as e:
                    attempt_count += 1
                    
            if not valid_edit:
                break
                
        return current_molecule
        
    def validate_molecule(self, molecule_string):
        # Input: molecule_string (SMILES)
        # Output: bool (True if valid, False otherwise)
        # Feel free to implement your own molecule validation
        raise NotImplementedError("Feel free to implement your own molecule validation")
        
    def generate_possible_edits(self, current_molecule, target_properties):
        # Input: current_molecule (SMILES), target_properties (dict)
        # Output: list of edit dicts, e.g., [{'instruction': 'edit_command'}, ...]
        # Feel free to implement your own edit generation
        raise NotImplementedError("Feel free to implement your own edit generation")

class MCTSMolecularOptimizer:
    def __init__(self, mcts_iterations, max_edit_count, exploration_constant,
                 expansion_depth, visit_threshold, batch_size, early_termination,
                 temperature_schedule, node_selection_method, backup_method,
                 simulation_policy, expansion_policy, final_selection_method,
                 convergence_threshold, max_simulation_depth, rollout_budget,
                 progressive_widening_factor, tree_reuse_policy):
        
        if mcts_iterations is None:
            raise ValueError("mcts_iterations is required")
        if max_edit_count is None:
            raise ValueError("max_edit_count is required")
        if exploration_constant is None:
            raise ValueError("exploration_constant is required")
        if expansion_depth is None:
            raise ValueError("expansion_depth is required")
        if visit_threshold is None:
            raise ValueError("visit_threshold is required")
        if batch_size is None:
            raise ValueError("batch_size is required")
        if early_termination is None:
            raise ValueError("early_termination is required")
        if not temperature_schedule:
            raise ValueError("temperature_schedule is required")
        if not node_selection_method:
            raise ValueError("node_selection_method is required")
        if not backup_method:
            raise ValueError("backup_method is required")
        if not simulation_policy:
            raise ValueError("simulation_policy is required")
        if not expansion_policy:
            raise ValueError("expansion_policy is required")
        if not final_selection_method:
            raise ValueError("final_selection_method is required")
        if convergence_threshold is None:
            raise ValueError("convergence_threshold is required")
        if max_simulation_depth is None:
            raise ValueError("max_simulation_depth is required")
        if rollout_budget is None:
            raise ValueError("rollout_budget is required")
        if progressive_widening_factor is None:
            raise ValueError("progressive_widening_factor is required")
        if not tree_reuse_policy:
            raise ValueError("tree_reuse_policy is required")
            
        self.mcts_iterations = mcts_iterations
        self.max_edit_count = max_edit_count
        self.exploration_constant = exploration_constant
        self.expansion_depth = expansion_depth
        self.visit_threshold = visit_threshold
        self.batch_size = batch_size
        self.early_termination = early_termination
        self.temperature_schedule = temperature_schedule
        self.node_selection_method = node_selection_method
        self.backup_method = backup_method
        self.simulation_policy = simulation_policy
        self.expansion_policy = expansion_policy
        self.final_selection_method = final_selection_method
        self.convergence_threshold = convergence_threshold
        self.max_simulation_depth = max_simulation_depth
        self.rollout_budget = rollout_budget
        self.progressive_widening_factor = progressive_widening_factor
        self.tree_reuse_policy = tree_reuse_policy
        
        self.property_calculator = None
        self.reward_calculator = None
        self.molecular_editor = None
        self.root_node = None
        
    def initialize_components(self, property_calc, reward_calc, mol_editor):
        self.property_calculator = property_calc
        self.reward_calculator = reward_calc
        self.molecular_editor = mol_editor
        
    def selection_phase(self, current_node):
        path = []
        
        while current_node.children and current_node.is_expanded:
            temperature = self.get_temperature(len(path))
            
            best_child = current_node.select_best_child(
                self.exploration_constant,
                temperature,
                0.0,
                self.node_selection_method,
                0.1
            )
            
            if best_child is None:
                break
                
            path.append(current_node)
            current_node = best_child
            
        return current_node, path
        
    def expansion_phase(self, leaf_node):
        if leaf_node.is_terminal or len(leaf_node.edit_history) >= self.max_edit_count:
            leaf_node.is_terminal = True
            return leaf_node
            
        if leaf_node.visit_count >= self.visit_threshold and not leaf_node.is_expanded:
            possible_edits = self.molecular_editor.generate_possible_edits(
                leaf_node.molecular_state, {}
            )
            
            max_children = int(self.progressive_widening_factor * math.sqrt(leaf_node.visit_count))
            selected_edits = possible_edits[:max_children]
            
            for edit in selected_edits:
                new_edit_history = leaf_node.edit_history + [edit]
                new_molecule = self.molecular_editor.apply_edit_sequence(
                    self.root_node.molecular_state, new_edit_history
                )
                
                child_node = MCTSNode(
                    molecular_state=new_molecule,
                    edit_history=new_edit_history,
                    parent_node=leaf_node,
                    action_taken=edit,
                    visit_threshold=self.visit_threshold,
                    expansion_depth=self.expansion_depth,
                    state_encoding_method="smiles"
                )
                
                leaf_node.children.append(child_node)
                
            leaf_node.is_expanded = True
            
            if leaf_node.children:
                return random.choice(leaf_node.children)
                
        return leaf_node
        
    def simulation_phase(self, node):
        simulation_molecule = node.molecular_state
        simulation_edits = node.edit_history.copy()
        simulation_depth = 0
        
        while (simulation_depth < self.max_simulation_depth and 
               len(simulation_edits) < self.max_edit_count):
            
            possible_edits = self.molecular_editor.generate_possible_edits(
                simulation_molecule, {}
            )
            
            if not possible_edits:
                break
                
            if self.simulation_policy == "random":
                selected_edit = random.choice(possible_edits)
            elif self.simulation_policy == "greedy":
                best_edit = None
                best_reward = -float('inf')
                
                current_props = self.property_calculator.calculate_all_properties(simulation_molecule)
                
                for edit in possible_edits:
                    test_edits = simulation_edits + [edit]
                    test_molecule = self.molecular_editor.apply_edit_sequence(
                        self.root_node.molecular_state, test_edits
                    )
                    test_props = self.property_calculator.calculate_all_properties(test_molecule)
                    test_reward = self.reward_calculator.calculate_total_reward(
                        test_props, current_props, 0
                    )
                    
                    if test_reward > best_reward:
                        best_reward = test_reward
                        best_edit = edit
                        
                selected_edit = best_edit if best_edit else random.choice(possible_edits)
            else:
                selected_edit = random.choice(possible_edits)
                
            simulation_edits.append(selected_edit)
            simulation_molecule = self.molecular_editor.apply_edit_sequence(
                self.root_node.molecular_state, simulation_edits
            )
            simulation_depth += 1
            
        final_properties = self.property_calculator.calculate_all_properties(simulation_molecule)
        initial_properties = self.property_calculator.calculate_all_properties(
            self.root_node.molecular_state
        )
        
        simulation_reward = self.reward_calculator.calculate_total_reward(
            final_properties, initial_properties, 0
        )
        
        return simulation_reward
        
    def backup_phase(self, path, leaf_node, reward):
        all_nodes = path + [leaf_node]
        
        for node in reversed(all_nodes):
            if self.backup_method == "average":
                node.update_statistics(reward, 0.95, 0.1)
            elif self.backup_method == "max":
                if reward > node.average_reward:
                    node.update_statistics(reward, 0.95, 0.1)
            else:
                node.update_statistics(reward, 0.95, 0.1)
                
    def get_temperature(self, depth):
        if self.temperature_schedule == "constant":
            return 1.0
        elif self.temperature_schedule == "linear_decay":
            return max(0.1, 1.0 - depth * 0.1)
        elif self.temperature_schedule == "exponential_decay":
            return math.exp(-depth * 0.2)
        else:
            return 1.0
            
    def optimize_molecule(self, initial_molecule):
        self.root_node = MCTSNode(
            molecular_state=initial_molecule,
            edit_history=[],
            parent_node=None,
            action_taken=None,
            visit_threshold=self.visit_threshold,
            expansion_depth=self.expansion_depth,
            state_encoding_method="smiles"
        )
        
        best_molecule = initial_molecule
        best_reward = -float('inf')
        
        for iteration in range(self.mcts_iterations):
            selected_node, path = self.selection_phase(self.root_node)
            expanded_node = self.expansion_phase(selected_node)
            simulation_reward = self.simulation_phase(expanded_node)
            self.backup_phase(path, expanded_node, simulation_reward)
            
            if simulation_reward > best_reward:
                best_reward = simulation_reward
                best_molecule = expanded_node.molecular_state
                
            if self.early_termination:
                current_props = self.property_calculator.calculate_all_properties(best_molecule)
                current_reward = self.reward_calculator.calculate_total_reward(current_props, None, 0)
                if current_reward > 5.0:
                    print(f"Early termination at iteration {iteration}")
                    break
                    
        if self.final_selection_method == "best_reward":
            return best_molecule
        elif self.final_selection_method == "most_visited":
            best_child = max(self.root_node.children, key=lambda x: x.visit_count)
            return best_child.molecular_state
        else:
            return best_molecule
            

def main():
    parser = argparse.ArgumentParser(description='Experimental MCTS-based molecular optimization with SpaRE steering')
    parser.add_argument('--property_config_file', required=True, help='Property configuration file')
    parser.add_argument('--calculation_backend_config', required=True, help='Calculation backend config')
    parser.add_argument('--validation_rules_file', required=True, help='Validation rules file')
    parser.add_argument('--reward_config_file', required=True, help='Reward configuration file')
    parser.add_argument('--constraint_definitions_file', required=True, help='Constraint definitions file')
    parser.add_argument('--model_path', required=True, help='Model path')
    parser.add_argument('--tokenizer_path', required=True, help='Tokenizer path')
    parser.add_argument('--sae_model_path', required=True, help='SAE model path')
    parser.add_argument('--vec_file_path', required=True, help='Vector file path')
    parser.add_argument('--target_layer_name', required=True, help='Target layer name')
    parser.add_argument('--layer_names', required=True, help='JSON string of layer names')
    parser.add_argument('--device_id', type=int, required=True, help='Device ID')
    parser.add_argument('--torch_dtype_str', required=True, help='Torch dtype string')
    parser.add_argument('--model_revision', required=True, help='Model revision')
    parser.add_argument('--trust_remote_code_flag', required=True, help='Trust remote code flag')
    parser.add_argument('--cache_dir_path', required=True, help='Cache directory path')
    parser.add_argument('--steering_strength', type=float, required=True, help='Steering strength')
    parser.add_argument('--sae_normalization_method', required=True, help='SAE normalization method')
    parser.add_argument('--intervention_type', required=True, help='Intervention type')
    parser.add_argument('--generation_params_json', required=True, help='Generation parameters JSON')
    parser.add_argument('--edit_template', required=True, help='Edit template')
    parser.add_argument('--molecule_start_token', required=True, help='Molecule start token')
    parser.add_argument('--molecule_end_token', required=True, help='Molecule end token')
    parser.add_argument('--max_edit_attempts', type=int, required=True, help='Max edit attempts')
    parser.add_argument('--edit_validation_method', required=True, help='Edit validation method')
    
    args = parser.parse_args()
    
    # Fixed MCTS parameters for "simplified" interface
    mcts_iterations = 500
    max_edit_count = 20
    exploration_constant = 1.414
    expansion_depth = 5
    visit_threshold = 10
    batch_size = 32
    early_termination = True
    temperature_schedule = "linear_decay"
    node_selection_method = "ucb"
    backup_method = "average"
    simulation_policy = "random"
    expansion_policy = "progressive_widening"
    final_selection_method = "best_reward"
    convergence_threshold = 0.01
    max_simulation_depth = 10
    rollout_budget = 100
    progressive_widening_factor = 1.0
    tree_reuse_policy = "none"
    
    layer_names = json.loads(args.layer_names)
    
    initial_molecule = input("Enter initial molecule SMILES: ")
    output_file = input("Enter output file path: ")
    
    prop_calc = MolecularPropertyCalculator(
        args.property_config_file, args.calculation_backend_config, args.validation_rules_file
    )
    
    reward_calc = RewardCalculator(
        args.reward_config_file, args.constraint_definitions_file
    )
    
    mol_editor = SpaREMolecularEditor(
        args.model_path, args.tokenizer_path, args.sae_model_path, args.vec_file_path,
        args.target_layer_name, layer_names, args.device_id, args.torch_dtype_str,
        args.model_revision, args.trust_remote_code_flag == "True", args.cache_dir_path,
        args.steering_strength, args.sae_normalization_method, args.intervention_type,
        args.generation_params_json, args.edit_template, args.molecule_start_token,
        args.molecule_end_token, args.max_edit_attempts, args.edit_validation_method
    )
    
    mol_editor.load_model_components()
    
    optimizer = MCTSMolecularOptimizer(
        mcts_iterations, max_edit_count, exploration_constant,
        expansion_depth, visit_threshold, batch_size, early_termination,
        temperature_schedule, node_selection_method, backup_method,
        simulation_policy, expansion_policy, final_selection_method,
        convergence_threshold, max_simulation_depth, rollout_budget,
        progressive_widening_factor, tree_reuse_policy
    )
    
    optimizer.initialize_components(prop_calc, reward_calc, mol_editor)
    
    optimized_molecule = optimizer.optimize_molecule(initial_molecule)
    
    with open(output_file, 'w') as f:
        f.write(optimized_molecule)
        
    print(f"Optimization complete. Result saved to {output_file}")

if __name__ == "__main__":
    main()