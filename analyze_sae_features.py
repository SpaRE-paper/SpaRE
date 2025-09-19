import torch
import torch.nn as nn
import json
import os
import sys
import numpy as np
import argparse

class SAEFeatureAnalyzer:
    def __init__(self, sae_model_path, pos_data_path, neg_data_path, output_vec_path,
                 device_id, batch_size_limit, threshold_false, threshold_true,
                 torch_dtype_str, minimum_samples, feature_name_prefix,
                 statistical_method):
        
        if not sae_model_path:
            raise ValueError("sae_model_path is required")
        if not pos_data_path:
            raise ValueError("pos_data_path is required") 
        if not neg_data_path:
            raise ValueError("neg_data_path is required")
        if not output_vec_path:
            raise ValueError("output_vec_path is required")
        if device_id is None:
            raise ValueError("device_id is required")
        if not batch_size_limit:
            raise ValueError("batch_size_limit is required")
        if threshold_false is None:
            raise ValueError("threshold_false is required")
        if threshold_true is None:
            raise ValueError("threshold_true is required")
        if not torch_dtype_str:
            raise ValueError("torch_dtype_str is required")
        if not minimum_samples:
            raise ValueError("minimum_samples is required")
        if not feature_name_prefix:
            raise ValueError("feature_name_prefix is required")
        if not statistical_method:
            raise ValueError("statistical_method is required")
            
        self.sae_model_path = sae_model_path
        self.pos_data_path = pos_data_path
        self.neg_data_path = neg_data_path
        self.output_vec_path = output_vec_path
        self.device_id = device_id
        self.batch_size_limit = batch_size_limit
        self.threshold_false = threshold_false
        self.threshold_true = threshold_true
        self.torch_dtype_str = torch_dtype_str
        self.minimum_samples = minimum_samples
        self.feature_name_prefix = feature_name_prefix
        self.statistical_method = statistical_method
        
        self.device = torch.device(f"cuda:{device_id}")
        
        if torch_dtype_str == "float16":
            self.torch_dtype = torch.float16
        elif torch_dtype_str == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif torch_dtype_str == "float32":
            self.torch_dtype = torch.float32
        else:
            raise ValueError("Invalid torch_dtype_str")
    
    def load_sae_model(self):
        model_state = torch.load(self.sae_model_path, map_location=self.device)
        
        input_dim = None
        hidden_dim = None
        
        for key, value in model_state.items():
            if "encoder.weight" in key:
                input_dim = value.shape[1]
                hidden_dim = value.shape[0]
                break
                
        if input_dim is None or hidden_dim is None:
            raise ValueError("Could not determine model dimensions")
            
        from train_sae import SparseAutoEncoder, NormalizedReLU
        
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
        
        self.sae_model.load_state_dict(model_state)
        self.sae_model.eval()
        
    def load_data(self):
        self.pos_data = torch.load(self.pos_data_path, map_location='cpu')
        self.neg_data = torch.load(self.neg_data_path, map_location='cpu')
        
        if len(self.pos_data.shape) != 2:
            self.pos_data = self.pos_data.view(-1, self.pos_data.shape[-1])
        if len(self.neg_data.shape) != 2:
            self.neg_data = self.neg_data.view(-1, self.neg_data.shape[-1])
            
        print(f"Loaded {self.pos_data.shape[0]} positive samples")
        print(f"Loaded {self.neg_data.shape[0]} negative samples")
        
    def encode_batch(self, data):
        encoded_data = []
        
        for i in range(0, len(data), self.batch_size_limit):
            batch = data[i:i+self.batch_size_limit].to(self.device)
            
            with torch.no_grad():
                encoded = self.sae_model.encoder(batch)
                activated = self.sae_model.activation(encoded)
                
            encoded_data.append(activated.cpu())
            
        return torch.cat(encoded_data, dim=0)
    
    def analyze_features(self):
        pos_encoded = self.encode_batch(self.pos_data)
        neg_encoded = self.encode_batch(self.neg_data)
        
        print(f"Encoded positive data shape: {pos_encoded.shape}")
        print(f"Encoded negative data shape: {neg_encoded.shape}")
        
        num_features = pos_encoded.shape[1]
        qualifying_features = {}
        
        for feature_idx in range(num_features):
            pos_feature_values = pos_encoded[:, feature_idx]
            neg_feature_values = neg_encoded[:, feature_idx]
            
            pos_true_mask = pos_feature_values >= self.threshold_true
            pos_false_mask = (pos_feature_values >= 0) & (pos_feature_values < self.threshold_false)
            
            neg_true_mask = neg_feature_values >= self.threshold_true
            neg_false_mask = (neg_feature_values >= 0) & (neg_feature_values < self.threshold_false)
            
            pos_all_true = torch.all(pos_true_mask)
            neg_all_false = torch.all(neg_false_mask)
            
            if pos_all_true and neg_all_false:
                pos_qualifying_values = pos_feature_values[pos_true_mask]
                
                if len(pos_qualifying_values) >= self.minimum_samples:
                    if self.statistical_method == "mean":
                        feature_value = torch.mean(pos_qualifying_values).item()
                    elif self.statistical_method == "median":
                        feature_value = torch.median(pos_qualifying_values).item()
                    elif self.statistical_method == "max":
                        feature_value = torch.max(pos_qualifying_values).item()
                    elif self.statistical_method == "min":
                        feature_value = torch.min(pos_qualifying_values).item()
                    elif self.statistical_method == "std":
                        feature_value = torch.std(pos_qualifying_values).item()
                    else:
                        raise ValueError("Invalid statistical_method")
                        
                    qualifying_features[feature_idx] = feature_value
                    
        print(f"Found {len(qualifying_features)} qualifying features")
        
        result = {
            "name": self.feature_name_prefix
        }
        result.update(qualifying_features)
        
        with open(self.output_vec_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result

def main():
    parser = argparse.ArgumentParser(description='Analyze SAE features and generate steering vectors')
    parser.add_argument('--sae_model_path', required=True, help='Path to SAE model')
    parser.add_argument('--pos_data_path', required=True, help='Positive data path')
    parser.add_argument('--neg_data_path', required=True, help='Negative data path')
    parser.add_argument('--output_vec_path', required=True, help='Output vector file path')
    parser.add_argument('--device_id', type=int, required=True, help='Device ID')
    parser.add_argument('--batch_size_limit', type=int, required=True, help='Batch size limit')
    parser.add_argument('--threshold_false', type=float, required=True, help='False threshold')
    parser.add_argument('--threshold_true', type=float, required=True, help='True threshold')
    parser.add_argument('--torch_dtype_str', required=True, help='Torch dtype string')
    parser.add_argument('--minimum_samples', type=int, required=True, help='Minimum samples')
    parser.add_argument('--feature_name_prefix', required=True, help='Feature name prefix')
    parser.add_argument('--statistical_method', required=True, help='Statistical method')
    
    args = parser.parse_args()
    
    analyzer = SAEFeatureAnalyzer(
        args.sae_model_path, args.pos_data_path, args.neg_data_path, args.output_vec_path,
        args.device_id, args.batch_size_limit, args.threshold_false, args.threshold_true,
        args.torch_dtype_str, args.minimum_samples, args.feature_name_prefix,
        args.statistical_method
    )
    
    analyzer.load_sae_model()
    analyzer.load_data()
    result = analyzer.analyze_features()
    
    print(f"Analysis complete. Results saved to {args.output_vec_path}")
    print(f"Found {len(result) - 1} qualifying features")

if __name__ == "__main__":
    main()