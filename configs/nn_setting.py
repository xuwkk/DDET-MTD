"""
Settings for Neural Network
"""
import torch
import sys
from configs.config import sys_config

if sys_config['case_name'] == 'case14':

    nn_setting = {
        # Network Structure
        'sample_length': 6,
        'lattent_dim': 10,
        'no_layer': 3,
        'feature_size':68,
        
        # Training
        'epochs': 1000,
        'lr': 1e-3,
        'patience': 10,
        'delta': 0,
        'model_path': 'saved_model/case14/checkpoint_rnn.pt',
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        #'device': "cpu",
        'batch_size': 32,
        'lattent_weight': 0.0,
        
        'train_prop': 0.6,
        'valid_prop': 0.2,
        
        # Recover Setting
        'recover_lr': 5*1e-3,
        'beta_real': 0.1,
        'beta_imag': 0.1,
        'beta_mag': 100,
        'mode': 'pre', 
        'max_step_size': 1000,
        'min_step_size': 50
    
}
