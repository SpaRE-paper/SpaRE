# Controllable Molecule Generation via Sparse Representation Editing: An Interpretability-Driven Perspective

## Installation

```bash
git clone https://github.com/your-repo/SpaRE.git
cd SpaRE
pip install -r requirements.txt
```

## 📚 Core Components

The SpaRE framework consists of three main stages: representation extraction, sparse disentanglement, and controllable editing.

### Stage 1: Representation Extraction

#### 1. Basic Activation Collection (`collect_activations.py`)

Extract neural activations from language models during molecule generation for sparse autoencoder training.

```bash
python collect_activations.py \
    --model_path <MODEL_PATH> \
    --layer_names <LAYER_NAMES> \
    --target_layer_name <TARGET_LAYER_NAME> \
    --device_id <DEVICE_ID> \
    --batch_size_limit <BATCH_SIZE_LIMIT> \
    --sequence_length_max <SEQUENCE_LENGTH_MAX> \
    --output_file_path <OUTPUT_FILE_PATH> \
    --tokenizer_path <TOKENIZER_PATH> \
    --model_revision <MODEL_REVISION> \
    --trust_remote_code_flag <TRUST_REMOTE_CODE_FLAG> \
    --torch_dtype_str <TORCH_DTYPE_STR> \
    --cache_dir_path <CACHE_DIR_PATH> \
    --use_fast_tokenizer <USE_FAST_TOKENIZER> \
    --padding_side <PADDING_SIDE> \
    --truncation_side <TRUNCATION_SIDE> \
    --model_max_length <MODEL_MAX_LENGTH>
```

#### 2. Targeted Activation Collection (`collect_token_activations.py`)

Collect activations for specific molecule tokens to enable local control mechanisms.

```bash
python collect_token_activations.py \
    --model_path <MODEL_PATH> \
    --layer_names <LAYER_NAMES> \
    --target_layer_name <TARGET_LAYER_NAME> \
    --device_id <DEVICE_ID> \
    --batch_size_limit <BATCH_SIZE_LIMIT> \
    --sequence_length_max <SEQUENCE_LENGTH_MAX> \
    --tokenizer_path <TOKENIZER_PATH> \
    --model_revision <MODEL_REVISION> \
    --trust_remote_code_flag <TRUST_REMOTE_CODE_FLAG> \
    --torch_dtype_str <TORCH_DTYPE_STR> \
    --cache_dir_path <CACHE_DIR_PATH> \
    --use_fast_tokenizer <USE_FAST_TOKENIZER> \
    --padding_side <PADDING_SIDE> \
    --truncation_side <TRUNCATION_SIDE> \
    --model_max_length <MODEL_MAX_LENGTH> \
    --target_tokens_file <TARGET_TOKENS_FILE> \
    --pos_output_path <POS_OUTPUT_PATH> \
    --neg_output_path <NEG_OUTPUT_PATH> \
    --generation_max_length <GENERATION_MAX_LENGTH> \
    --temperature_val <TEMPERATURE_VAL> \
    --top_k_val <TOP_K_VAL> \
    --top_p_val <TOP_P_VAL> \
    --repetition_penalty_val <REPETITION_PENALTY_VAL> \
    --do_sample_flag <DO_SAMPLE_FLAG> \
    --num_beams <NUM_BEAMS> \
    --early_stopping_flag <EARLY_STOPPING_FLAG> \
    --pad_token_id_val <PAD_TOKEN_ID_VAL> \
    --eos_token_id_val <EOS_TOKEN_ID_VAL> \
    --seed_val <SEED_VAL>
```

#### 3. Property-Based Activation Collection (`collect_molecule_activations.py`)

Collect activations based on molecule property validation for global control mechanisms.

```bash
python collect_molecule_activations.py \
    --model_path <MODEL_PATH> \
    --layer_names <LAYER_NAMES> \
    --target_layer_name <TARGET_LAYER_NAME> \
    --device_id <DEVICE_ID> \
    --batch_size_limit <BATCH_SIZE_LIMIT> \
    --sequence_length_max <SEQUENCE_LENGTH_MAX> \
    --tokenizer_path <TOKENIZER_PATH> \
    --model_revision <MODEL_REVISION> \
    --trust_remote_code_flag <TRUST_REMOTE_CODE_FLAG> \
    --torch_dtype_str <TORCH_DTYPE_STR> \
    --cache_dir_path <CACHE_DIR_PATH> \
    --use_fast_tokenizer <USE_FAST_TOKENIZER> \
    --padding_side <PADDING_SIDE> \
    --truncation_side <TRUNCATION_SIDE> \
    --model_max_length <MODEL_MAX_LENGTH> \
    --pos_output_path <POS_OUTPUT_PATH> \
    --neg_output_path <NEG_OUTPUT_PATH> \
    --generation_max_length <GENERATION_MAX_LENGTH> \
    --temperature_val <TEMPERATURE_VAL> \
    --top_k_val <TOP_K_VAL> \
    --top_p_val <TOP_P_VAL> \
    --repetition_penalty_val <REPETITION_PENALTY_VAL> \
    --do_sample_flag <DO_SAMPLE_FLAG> \
    --num_beams <NUM_BEAMS> \
    --early_stopping_flag <EARLY_STOPPING_FLAG> \
    --pad_token_id_val <PAD_TOKEN_ID_VAL> \
    --eos_token_id_val <EOS_TOKEN_ID_VAL> \
    --seed_val <SEED_VAL> \
    --num_generations <NUM_GENERATIONS> \
    --validation_function_path <VALIDATION_FUNCTION_PATH> \
    --validation_module_name <VALIDATION_MODULE_NAME> \
    --validation_function_name <VALIDATION_FUNCTION_NAME> \
    --molecule_start_token <MOLECULE_START_TOKEN> \
    --molecule_end_token <MOLECULE_END_TOKEN>
```

### Stage 2: Sparse Disentanglement

#### 4. Sparse Autoencoder Training (`train_sae.py`)

Train sparse autoencoders with normalized ReLU to disentangle molecule representations into interpretable concepts.

```bash
python train_sae.py \
    --data_file_path <DATA_FILE_PATH> \
    --model_save_path <MODEL_SAVE_PATH> \
    --input_dim <INPUT_DIM> \
    --hidden_dim <HIDDEN_DIM> \
    --sparsity_coeff <SPARSITY_COEFF> \
    --learning_rate_encoder <LEARNING_RATE_ENCODER> \
    --learning_rate_decoder <LEARNING_RATE_DECODER> \
    --weight_decay_val <WEIGHT_DECAY_VAL> \
    --eps_val <EPS_VAL> \
    --normalization_type <NORMALIZATION_TYPE> \
    --momentum_val <MOMENTUM_VAL> \
    --affine_transform <AFFINE_TRANSFORM> \
    --track_running_stats <TRACK_RUNNING_STATS> \
    --bias_encoder <BIAS_ENCODER> \
    --bias_decoder <BIAS_DECODER> \
    --init_method <INIT_METHOD> \
    --init_scale <INIT_SCALE> \
    --num_epochs <NUM_EPOCHS> \
    --batch_size <BATCH_SIZE> \
    --l1_coeff <L1_COEFF> \
    --l2_coeff <L2_COEFF> \
    --reconstruction_loss_type <RECONSTRUCTION_LOSS_TYPE> \
    --reduction_type <REDUCTION_TYPE> \
    --device_id <DEVICE_ID> \
    --checkpoint_freq <CHECKPOINT_FREQ> \
    --log_freq <LOG_FREQ> \
    --validation_split <VALIDATION_SPLIT> \
    --shuffle_data <SHUFFLE_DATA> \
    --pin_memory <PIN_MEMORY> \
    --num_workers <NUM_WORKERS> \
    --drop_last <DROP_LAST> \
    --gradient_clip_val <GRADIENT_CLIP_VAL> \
    --scheduler_type <SCHEDULER_TYPE> \
    --scheduler_params <SCHEDULER_PARAMS> \
    --early_stopping_patience <EARLY_STOPPING_PATIENCE> \
    --min_delta <MIN_DELTA>
```

#### 5. Interpretable Feature Discovery (`analyze_sae_features.py`)

Analyze trained sparse autoencoders to identify and extract interpretable molecule concepts.

```bash
python analyze_sae_features.py \
    --sae_model_path <SAE_MODEL_PATH> \
    --pos_data_path <POS_DATA_PATH> \
    --neg_data_path <NEG_DATA_PATH> \
    --output_vec_path <OUTPUT_VEC_PATH> \
    --device_id <DEVICE_ID> \
    --batch_size_limit <BATCH_SIZE_LIMIT> \
    --threshold_false <THRESHOLD_FALSE> \
    --threshold_true <THRESHOLD_TRUE> \
    --torch_dtype_str <TORCH_DTYPE_STR> \
    --minimum_samples <MINIMUM_SAMPLES> \
    --feature_name_prefix <FEATURE_NAME_PREFIX> \
    --statistical_method <STATISTICAL_METHOD>
```

### Stage 3: Controllable Editing

#### 6. Local Control (`steer_nth_token.py`)

Apply precise editing at specific molecule positions for local control over atoms and functional groups.

```bash
python steer_nth_token.py \
    --model_path <MODEL_PATH> \
    --sae_model_path <SAE_MODEL_PATH> \
    --vec_file_path <VEC_FILE_PATH> \
    --target_layer_name <TARGET_LAYER_NAME> \
    --layer_names <LAYER_NAMES> \
    --device_id <DEVICE_ID> \
    --torch_dtype_str <TORCH_DTYPE_STR> \
    --tokenizer_path <TOKENIZER_PATH> \
    --model_revision <MODEL_REVISION> \
    --trust_remote_code_flag <TRUST_REMOTE_CODE_FLAG> \
    --cache_dir_path <CACHE_DIR_PATH> \
    --use_fast_tokenizer <USE_FAST_TOKENIZER> \
    --padding_side <PADDING_SIDE> \
    --model_max_length <MODEL_MAX_LENGTH> \
    --target_token_position <TARGET_TOKEN_POSITION> \
    --steering_strength <STEERING_STRENGTH> \
    --normalization_method <NORMALIZATION_METHOD> \
    --normalization_eps <NORMALIZATION_EPS> \
    --intervention_type <INTERVENTION_TYPE> \
    --gradient_checkpointing <GRADIENT_CHECKPOINTING> \
    --debug_mode <DEBUG_MODE> \
    --output_file_path <OUTPUT_FILE_PATH>
```

#### 7. Global Control (`steer_full_sequence.py`)

Apply concept-based editing across molecule sequences for global control over structural and physicochemical properties.

```bash
python steer_full_sequence.py \
    --model_path <MODEL_PATH> \
    --sae_model_path <SAE_MODEL_PATH> \
    --vec_file_path <VEC_FILE_PATH> \
    --target_layer_name <TARGET_LAYER_NAME> \
    --layer_names <LAYER_NAMES> \
    --device_id <DEVICE_ID> \
    --torch_dtype_str <TORCH_DTYPE_STR> \
    --tokenizer_path <TOKENIZER_PATH> \
    --model_revision <MODEL_REVISION> \
    --trust_remote_code_flag <TRUST_REMOTE_CODE_FLAG> \
    --cache_dir_path <CACHE_DIR_PATH> \
    --use_fast_tokenizer <USE_FAST_TOKENIZER> \
    --padding_side <PADDING_SIDE> \
    --model_max_length <MODEL_MAX_LENGTH> \
    --steering_strength <STEERING_STRENGTH> \
    --normalization_method <NORMALIZATION_METHOD> \
    --normalization_eps <NORMALIZATION_EPS> \
    --intervention_type <INTERVENTION_TYPE> \
    --gradient_checkpointing <GRADIENT_CHECKPOINTING> \
    --debug_mode <DEBUG_MODE> \
    --output_file_path <OUTPUT_FILE_PATH> \
    --steering_schedule <STEERING_SCHEDULE> \
    --position_dependent_strength <POSITION_DEPENDENT_STRENGTH> \
    --decay_factor <DECAY_FACTOR> \
    --warmup_steps <WARMUP_STEPS> \
    --cooldown_steps <COOLDOWN_STEPS> \
    --sequence_chunking_size <SEQUENCE_CHUNKING_SIZE>
```

## 🔬 Experimental Tools

### MCTS Optimizer (`mcts_molecular_optimizer.py`)

Experimental multi-step optimization using Monte Carlo Tree Search. This is a research prototype demonstrating advanced applications of the SpaRE framework.

*Note: This is experimental code for research purposes and not part of the core toolkit.*

### Dual-Token Position Sweep (`dual_token_sweep_steering.py`)

Experimental systematic position-wise intervention sweep using two different SAE models and steering vectors.

```bash
python dual_token_sweep_steering.py \
    --model_path <MODEL_PATH> \
    --sae_model_path_1 <SAE_MODEL_PATH_1> \
    --sae_model_path_2 <SAE_MODEL_PATH_2> \
    --vec_file_path_1 <VEC_FILE_PATH_1> \
    --vec_file_path_2 <VEC_FILE_PATH_2> \
    --target_layer_name_1 <TARGET_LAYER_NAME_1> \
    --target_layer_name_2 <TARGET_LAYER_NAME_2> \
    --layer_names <LAYER_NAMES> \
    --device_id <DEVICE_ID> \
    --torch_dtype_str <TORCH_DTYPE_STR> \
    --tokenizer_path <TOKENIZER_PATH> \
    --model_revision <MODEL_REVISION> \
    --trust_remote_code_flag <TRUST_REMOTE_CODE_FLAG> \
    --cache_dir_path <CACHE_DIR_PATH> \
    --use_fast_tokenizer <USE_FAST_TOKENIZER> \
    --padding_side <PADDING_SIDE> \
    --model_max_length <MODEL_MAX_LENGTH> \
    --steering_strength_1 <STEERING_STRENGTH_1> \
    --steering_strength_2 <STEERING_STRENGTH_2> \
    --normalization_method_1 <NORMALIZATION_METHOD_1> \
    --normalization_method_2 <NORMALIZATION_METHOD_2> \
    --normalization_eps_1 <NORMALIZATION_EPS_1> \
    --normalization_eps_2 <NORMALIZATION_EPS_2> \
    --intervention_type_1 <INTERVENTION_TYPE_1> \
    --intervention_type_2 <INTERVENTION_TYPE_2> \
    --gradient_checkpointing <GRADIENT_CHECKPOINTING> \
    --debug_mode <DEBUG_MODE> \
    --sweep_output_dir <SWEEP_OUTPUT_DIR> \
    --baseline_output_file <BASELINE_OUTPUT_FILE> \
    --generation_max_length <GENERATION_MAX_LENGTH> \
    --temperature_val <TEMPERATURE_VAL> \
    --top_k_val <TOP_K_VAL> \
    --top_p_val <TOP_P_VAL> \
    --repetition_penalty_val <REPETITION_PENALTY_VAL> \
    --do_sample_flag <DO_SAMPLE_FLAG> \
    --num_beams <NUM_BEAMS> \
    --early_stopping_flag <EARLY_STOPPING_FLAG> \
    --pad_token_id_val <PAD_TOKEN_ID_VAL> \
    --eos_token_id_val <EOS_TOKEN_ID_VAL> \
    --seed_val <SEED_VAL> \
    --max_sweep_iterations <MAX_SWEEP_ITERATIONS> \
    --min_position_gap <MIN_POSITION_GAP> \
    --sweep_mode <SWEEP_MODE> \
    --position_selection_strategy <POSITION_SELECTION_STRATEGY> \
    --overlap_handling_method <OVERLAP_HANDLING_METHOD>
```

*Note: This is experimental code for research purposes and not part of the core toolkit.*

## 🎯 Customization Framework

For maximum adaptability to different experimental setups, we have defined several standardized functions that users can freely implement according to their specific needs:

### Molecule Validation Functions
The framework includes flexible validation interfaces for different control paradigms:

**Basic Validation (`collect_molecule_activations.py`)**:
- `molecule_validator(generated_string)` - Define your molecule validation criteria
  - Input: generated_string (str) - molecule string  
  - Output: bool - True if meets requirements, False otherwise
  - Feel free to implement your own validation logic

### Advanced Research Extensions

The experimental MCTS optimizer (`mcts_molecular_optimizer.py`) provides additional customization points for advanced molecule optimization research:

**Property Calculation Interface**:
- `MolecularPropertyCalculator.__init__(property_config_file, calculation_backend_config, validation_rules_file)` - Initialize your property calculation system
- `calculate_all_properties` - Implement your molecule property pipeline
  - Input: molecule (str)
  - Output: dict with keys 'logp', 'mw', 'tpsa', 'hbd', 'hba', 'aromatic_rings', 'sa_score', 'qed'
  - Feel free to implement your own property calculation

**Reward Function Interface**:
- `RewardCalculator.__init__(reward_config_file, constraint_definitions_file)` - Initialize your reward system
- `calculate_total_reward(current_properties, previous_properties, consecutive_violations)` - Design your optimization objective
  - Input: current_properties (dict), previous_properties (dict or None), consecutive_violations (int)
  - Output: float reward value
  - Feel free to implement your own reward calculation

**Molecular State Analysis**:
- `calculate_state_similarity(state1, state2)` - Define molecule similarity metrics
  - Input: state1, state2 (molecule strings)
  - Output: float similarity score (0.0 to 1.0)
  - Feel free to implement your own similarity calculation

**Molecular Editing Strategy**:
- `validate_molecule(molecule_string)` - Validate generated molecules
  - Input: molecule_string
  - Output: bool (True if valid, False otherwise)
  - Feel free to implement your own molecule validation

- `generate_possible_edits(current_molecule, target_properties)` - Generate molecule transformation strategies
  - Input: current_molecule, target_properties (dict)
  - Output: list of edit dicts, e.g., [{'instruction': 'edit_command'}, ...]
  - Feel free to implement your own edit generation

These interfaces enable researchers to adapt SpaRE to diverse molecule domains and optimization objectives while maintaining the core interpretability-driven framework.

## 📖 Quick Start Guide

### Typical Workflow:

1. **Representation Extraction**: Collect neural activations from your molecule generation model
2. **Sparse Disentanglement**: Train sparse autoencoders to discover interpretable concepts  
3. **Feature Analysis**: Identify meaningful molecule features for control
4. **Controllable Generation**: Apply local or global editing for desired molecule properties

### Local Control Example:
```bash
# Collect activations for specific tokens
python collect_token_activations.py [args...]

# Apply position-specific control
python steer_nth_token.py [args...]
```

### Global Control Example:
```bash
# Collect property-based activations  
python collect_molecule_activations.py [args...]

# Apply global property control
python steer_full_sequence.py [args...]
```


## 🤝 Contributing

We welcome contributions! The modular design makes it easy to extend functionality. Feel free to submit pull requests with your improvements.

## 📄 License

This project is licensed under the GNU General Public License v2.0 - see the LICENSE file for details.

## 🙏 Acknowledgments

We thank the reviewers and the research community for their valuable feedback. We're excited to see how you'll use SpaRE to advance controllable molecule generation!

---

For questions and support, please open an issue.