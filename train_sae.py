import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
import json
import math
import argparse

class NormalizedReLU(nn.Module):
    def __init__(self, dim, eps, normalization_type, momentum_val, affine_transform, track_running_stats):
        super().__init__()
        if eps is None:
            raise ValueError("eps is required")
        if not normalization_type:
            raise ValueError("normalization_type is required")
        if momentum_val is None:
            raise ValueError("momentum_val is required")
        if affine_transform is None:
            raise ValueError("affine_transform is required")
        if track_running_stats is None:
            raise ValueError("track_running_stats is required")
            
        self.relu = nn.ReLU()
        if normalization_type == "layer":
            self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=affine_transform)
        elif normalization_type == "batch":
            self.norm = nn.BatchNorm1d(dim, eps=eps, momentum=momentum_val, 
                                     affine=affine_transform, track_running_stats=track_running_stats)
        else:
            raise ValueError("Invalid normalization_type")
            
    def forward(self, x):
        x = self.relu(x)
        if len(x.shape) == 2 and hasattr(self.norm, 'num_features'):
            x = self.norm(x)
        elif len(x.shape) == 3 and hasattr(self.norm, 'normalized_shape'):
            x = self.norm(x)
        else:
            x = self.norm(x)
        return x

class SparseAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_coeff, learning_rate_encoder, 
                 learning_rate_decoder, weight_decay_val, eps_val, normalization_type,
                 momentum_val, affine_transform, track_running_stats, bias_encoder,
                 bias_decoder, init_method, init_scale):
        super().__init__()
        
        if not input_dim:
            raise ValueError("input_dim is required")
        if not hidden_dim:
            raise ValueError("hidden_dim is required")
        if sparsity_coeff is None:
            raise ValueError("sparsity_coeff is required")
        if learning_rate_encoder is None:
            raise ValueError("learning_rate_encoder is required")
        if learning_rate_decoder is None:
            raise ValueError("learning_rate_decoder is required")
        if weight_decay_val is None:
            raise ValueError("weight_decay_val is required")
        if eps_val is None:
            raise ValueError("eps_val is required")
        if not normalization_type:
            raise ValueError("normalization_type is required")
        if momentum_val is None:
            raise ValueError("momentum_val is required")
        if affine_transform is None:
            raise ValueError("affine_transform is required")
        if track_running_stats is None:
            raise ValueError("track_running_stats is required")
        if bias_encoder is None:
            raise ValueError("bias_encoder is required")
        if bias_decoder is None:
            raise ValueError("bias_decoder is required")
        if not init_method:
            raise ValueError("init_method is required")
        if init_scale is None:
            raise ValueError("init_scale is required")
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coeff = sparsity_coeff
        
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=bias_encoder)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=bias_decoder)
        self.activation = NormalizedReLU(hidden_dim, eps_val, normalization_type, 
                                       momentum_val, affine_transform, track_running_stats)
        
        self.init_weights(init_method, init_scale)
        
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), 
                                          lr=learning_rate_encoder, weight_decay=weight_decay_val)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), 
                                          lr=learning_rate_decoder, weight_decay=weight_decay_val)
        self.activation_optimizer = optim.Adam(self.activation.parameters(), 
                                             lr=learning_rate_encoder, weight_decay=weight_decay_val)
        
    def init_weights(self, init_method, init_scale):
        if init_method == "xavier_uniform":
            nn.init.xavier_uniform_(self.encoder.weight, gain=init_scale)
            nn.init.xavier_uniform_(self.decoder.weight, gain=init_scale)
        elif init_method == "xavier_normal":
            nn.init.xavier_normal_(self.encoder.weight, gain=init_scale)
            nn.init.xavier_normal_(self.decoder.weight, gain=init_scale)
        elif init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.encoder.weight, a=init_scale)
            nn.init.kaiming_uniform_(self.decoder.weight, a=init_scale)
        elif init_method == "kaiming_normal":
            nn.init.kaiming_normal_(self.encoder.weight, a=init_scale)
            nn.init.kaiming_normal_(self.decoder.weight, a=init_scale)
        elif init_method == "normal":
            nn.init.normal_(self.encoder.weight, mean=0, std=init_scale)
            nn.init.normal_(self.decoder.weight, mean=0, std=init_scale)
        else:
            raise ValueError("Invalid init_method")
            
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)
        
    def forward(self, x):
        encoded = self.encoder(x)
        activated = self.activation(encoded)
        decoded = self.decoder(activated)
        return decoded, activated
        
    def compute_loss(self, x, reconstruction, activations, l1_coeff, l2_coeff, 
                    reconstruction_loss_type, reduction_type):
        if not reconstruction_loss_type:
            raise ValueError("reconstruction_loss_type is required")
        if not reduction_type:
            raise ValueError("reduction_type is required")
        if l1_coeff is None:
            raise ValueError("l1_coeff is required")
        if l2_coeff is None:
            raise ValueError("l2_coeff is required")
            
        if reconstruction_loss_type == "mse":
            recon_loss = nn.MSELoss(reduction=reduction_type)(reconstruction, x)
        elif reconstruction_loss_type == "l1":
            recon_loss = nn.L1Loss(reduction=reduction_type)(reconstruction, x)
        elif reconstruction_loss_type == "huber":
            recon_loss = nn.SmoothL1Loss(reduction=reduction_type)(reconstruction, x)
        else:
            raise ValueError("Invalid reconstruction_loss_type")
            
        l1_penalty = l1_coeff * torch.mean(torch.abs(activations))
        l2_penalty = l2_coeff * torch.mean(activations ** 2)
        sparsity_penalty = self.sparsity_coeff * torch.mean(torch.abs(activations))
        
        total_loss = recon_loss + l1_penalty + l2_penalty + sparsity_penalty
        
        return total_loss, recon_loss, l1_penalty, l2_penalty, sparsity_penalty

def train_sae(data_file_path, model_save_path, input_dim, hidden_dim, sparsity_coeff, 
              learning_rate_encoder, learning_rate_decoder, weight_decay_val, eps_val,
              normalization_type, momentum_val, affine_transform, track_running_stats,
              bias_encoder, bias_decoder, init_method, init_scale, num_epochs,
              batch_size, l1_coeff, l2_coeff, reconstruction_loss_type, reduction_type,
              device_id, checkpoint_freq, log_freq, validation_split, shuffle_data,
              pin_memory, num_workers, drop_last, gradient_clip_val, scheduler_type,
              scheduler_params, early_stopping_patience, min_delta):
    
    if not data_file_path:
        raise ValueError("data_file_path is required")
    if not model_save_path:
        raise ValueError("model_save_path is required")
    if not num_epochs:
        raise ValueError("num_epochs is required")
    if not batch_size:
        raise ValueError("batch_size is required")
    if device_id is None:
        raise ValueError("device_id is required")
    if not checkpoint_freq:
        raise ValueError("checkpoint_freq is required")
    if not log_freq:
        raise ValueError("log_freq is required")
    if validation_split is None:
        raise ValueError("validation_split is required")
    if shuffle_data is None:
        raise ValueError("shuffle_data is required")
    if pin_memory is None:
        raise ValueError("pin_memory is required")
    if not num_workers:
        raise ValueError("num_workers is required")
    if drop_last is None:
        raise ValueError("drop_last is required")
    if gradient_clip_val is None:
        raise ValueError("gradient_clip_val is required")
    if not scheduler_type:
        raise ValueError("scheduler_type is required")
    if not scheduler_params:
        raise ValueError("scheduler_params is required")
    if early_stopping_patience is None:
        raise ValueError("early_stopping_patience is required")
    if min_delta is None:
        raise ValueError("min_delta is required")
    
    device = torch.device(f"cuda:{device_id}")
    
    data = torch.load(data_file_path)
    if len(data.shape) != 2:
        data = data.view(-1, data.shape[-1])
        
    dataset_size = len(data)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_data,
                            pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last)
    
    model = SparseAutoEncoder(input_dim, hidden_dim, sparsity_coeff, learning_rate_encoder,
                            learning_rate_decoder, weight_decay_val, eps_val, normalization_type,
                            momentum_val, affine_transform, track_running_stats, bias_encoder,
                            bias_decoder, init_method, init_scale).to(device)
    
    scheduler_params_dict = json.loads(scheduler_params)
    
    if scheduler_type == "step":
        encoder_scheduler = optim.lr_scheduler.StepLR(model.encoder_optimizer, **scheduler_params_dict)
        decoder_scheduler = optim.lr_scheduler.StepLR(model.decoder_optimizer, **scheduler_params_dict)
        activation_scheduler = optim.lr_scheduler.StepLR(model.activation_optimizer, **scheduler_params_dict)
    elif scheduler_type == "cosine":
        encoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(model.encoder_optimizer, **scheduler_params_dict)
        decoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(model.decoder_optimizer, **scheduler_params_dict)
        activation_scheduler = optim.lr_scheduler.CosineAnnealingLR(model.activation_optimizer, **scheduler_params_dict)
    else:
        raise ValueError("Invalid scheduler_type")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (batch_data,) in enumerate(train_loader):
            batch_data = batch_data.to(device)
            
            model.encoder_optimizer.zero_grad()
            model.decoder_optimizer.zero_grad()
            model.activation_optimizer.zero_grad()
            
            reconstruction, activations = model(batch_data)
            loss, recon_loss, l1_penalty, l2_penalty, sparsity_penalty = model.compute_loss(
                batch_data, reconstruction, activations, l1_coeff, l2_coeff,
                reconstruction_loss_type, reduction_type
            )
            
            loss.backward()
            
            if gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            
            model.encoder_optimizer.step()
            model.decoder_optimizer.step()
            model.activation_optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % log_freq == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        encoder_scheduler.step()
        decoder_scheduler.step()
        activation_scheduler.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data, in val_loader:
                batch_data = batch_data.to(device)
                reconstruction, activations = model(batch_data)
                loss, _, _, _, _ = model.compute_loss(
                    batch_data, reconstruction, activations, l1_coeff, l2_coeff,
                    reconstruction_loss_type, reduction_type
                )
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
        if epoch % checkpoint_freq == 0:
            checkpoint_path = f"{model_save_path}_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)

def main():
    parser = argparse.ArgumentParser(description='Train sparse autoencoder for interpretable molecular generation')
    parser.add_argument('--data_file_path', required=True, help='Path to training data file')
    parser.add_argument('--model_save_path', required=True, help='Path to save trained model')
    parser.add_argument('--input_dim', type=int, required=True, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, required=True, help='Hidden dimension')
    parser.add_argument('--sparsity_coeff', type=float, required=True, help='Sparsity coefficient')
    parser.add_argument('--learning_rate_encoder', type=float, required=True, help='Learning rate for encoder')
    parser.add_argument('--learning_rate_decoder', type=float, required=True, help='Learning rate for decoder')
    parser.add_argument('--weight_decay_val', type=float, required=True, help='Weight decay value')
    parser.add_argument('--eps_val', type=float, required=True, help='Epsilon value')
    parser.add_argument('--normalization_type', required=True, help='Normalization type')
    parser.add_argument('--momentum_val', type=float, required=True, help='Momentum value')
    parser.add_argument('--affine_transform', required=True, help='Affine transform flag')
    parser.add_argument('--track_running_stats', required=True, help='Track running stats flag')
    parser.add_argument('--bias_encoder', required=True, help='Encoder bias flag')
    parser.add_argument('--bias_decoder', required=True, help='Decoder bias flag')
    parser.add_argument('--init_method', required=True, help='Initialization method')
    parser.add_argument('--init_scale', type=float, required=True, help='Initialization scale')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--l1_coeff', type=float, required=True, help='L1 coefficient')
    parser.add_argument('--l2_coeff', type=float, required=True, help='L2 coefficient')
    parser.add_argument('--reconstruction_loss_type', required=True, help='Reconstruction loss type')
    parser.add_argument('--reduction_type', required=True, help='Reduction type')
    parser.add_argument('--device_id', type=int, required=True, help='Device ID')
    parser.add_argument('--checkpoint_freq', type=int, required=True, help='Checkpoint frequency')
    parser.add_argument('--log_freq', type=int, required=True, help='Log frequency')
    parser.add_argument('--validation_split', type=float, required=True, help='Validation split')
    parser.add_argument('--shuffle_data', required=True, help='Shuffle data flag')
    parser.add_argument('--pin_memory', required=True, help='Pin memory flag')
    parser.add_argument('--num_workers', type=int, required=True, help='Number of workers')
    parser.add_argument('--drop_last', required=True, help='Drop last flag')
    parser.add_argument('--gradient_clip_val', type=float, required=True, help='Gradient clip value')
    parser.add_argument('--scheduler_type', required=True, help='Scheduler type')
    parser.add_argument('--scheduler_params', required=True, help='Scheduler parameters')
    parser.add_argument('--early_stopping_patience', type=int, required=True, help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, required=True, help='Minimum delta')
    
    args = parser.parse_args()
    
    train_sae(
        args.data_file_path, args.model_save_path, args.input_dim, args.hidden_dim, args.sparsity_coeff,
        args.learning_rate_encoder, args.learning_rate_decoder, args.weight_decay_val, args.eps_val,
        args.normalization_type, args.momentum_val, args.affine_transform == "True", args.track_running_stats == "True",
        args.bias_encoder == "True", args.bias_decoder == "True", args.init_method, args.init_scale,
        args.num_epochs, args.batch_size, args.l1_coeff, args.l2_coeff,
        args.reconstruction_loss_type, args.reduction_type, args.device_id, args.checkpoint_freq, args.log_freq,
        args.validation_split, args.shuffle_data == "True", args.pin_memory == "True", args.num_workers,
        args.drop_last == "True", args.gradient_clip_val, args.scheduler_type, args.scheduler_params, args.early_stopping_patience,
        args.min_delta
    )

if __name__ == "__main__":
    main()