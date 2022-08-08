"""
Load the case class based on the specifications in sys_config
"""

from pypower.api import case14
import numpy as np
from configs.config import sys_config
from configs.config_mea_idx import define_mea_idx_noise
from gen_data.gen_data import gen_case
from utils.fdi_att import FDI
from models.dataset import rnn_dataset
from torch.utils.data import DataLoader
from models.dataset import rnn_dataset_evl

# Case Class
def load_case():
    """
    Return the instance case class
    """
    if sys_config['case_name'] == 'case14':
        case = case14()
    else:
        print("I have not written other cases!")
        
    case = gen_case(sys_config['case_name']) # Modify case
    # Instance case class
    noise_sigma_dir = f'gen_data/{sys_config["case_name"]}/noise_sigma.npy'
    idx, no_mea, _ = define_mea_idx_noise(case, sys_config['measure_type'])
    noise_sigma = np.load(noise_sigma_dir)

    case_class = FDI(case, noise_sigma, idx, sys_config['fpr'])
    
    return case_class

# Load and PV

def load_load_pv():

    load_active_dir = f'gen_data/{sys_config["case_name"]}/load_active.npy'
    load_reactive_dir = f'gen_data/{sys_config["case_name"]}/load_reactive.npy'
    pv_active_dir = f'gen_data/{sys_config["case_name"]}/pv_active.npy'
    pv_reactive_dir = f'gen_data/{sys_config["case_name"]}/pv_reactive.npy'

    load_active = np.load(load_active_dir)
    load_reactive = np.load(load_reactive_dir)
    pv_active = np.load(pv_active_dir)
    pv_reactive = np.load(pv_reactive_dir)

    # set the length equal to the bus number
    pv_active_ = np.zeros((load_active.shape[0], load_reactive.shape[1]))
    pv_reactive_ = np.zeros((load_reactive.shape[0], load_reactive.shape[1]))
    pv_active_[:,sys_config['pv_bus']] = pv_active     
    pv_reactive_[:,sys_config['pv_bus']] = pv_reactive
    
    return load_active, load_reactive, pv_active_, pv_reactive_

def load_measurement():

    # Load measurement
    z_noise_summary = np.load('gen_data/case14/z_noise_summary.npy')     # Measurement with noise
    v_est_summary = np.load('gen_data/case14/v_est_summary.npy')          # Estimated voltage state
    success_summary = np.load('gen_data/case14/success_summary.npy')
    print(f'z noise size: {z_noise_summary.shape}')
    print(f'v est size: {v_est_summary.shape}')
    
    return z_noise_summary, v_est_summary

def load_dataset():

    is_shuffle = True  # Allow shuffle in dataloader
    
    # Test dataset dataloader: SCALED
    test_dataset_scaled = rnn_dataset(mode = 'test', istransform=True)
    test_dataloader_scaled = DataLoader(dataset = test_dataset_scaled,
                                    shuffle=is_shuffle,
                                    batch_size=1)
    print(f'test dataset size scaled: {len(test_dataloader_scaled.dataset)}') 

    # Test dataset dataloader: UNSCALED
    # With Random
    test_dataset_unscaled = rnn_dataset_evl(mode = 'test', istransform=False)
    test_dataloader_unscaled = DataLoader(dataset = test_dataset_unscaled,
                                shuffle=is_shuffle,
                                batch_size=1)
    print(f'test dataset size unscaled: {len(test_dataloader_unscaled.dataset)}')

    # Valid dataset: SCALED
    valid_dataset_scaled = rnn_dataset(mode = 'valid', istransform=True)
    valid_dataloader_scaled = DataLoader(dataset = valid_dataset_scaled,
                                    shuffle=is_shuffle,
                                    batch_size=1)
    print(f'valid dataset size scaled: {len(valid_dataloader_scaled.dataset)}') 

    # Validation dataset dataloader: UNSCALED
    # No Random
    valid_dataset_unscaled = rnn_dataset_evl(mode = 'valid', istransform=False)
    valid_dataloader_unscaled = DataLoader(dataset = valid_dataset_unscaled,
                                shuffle=is_shuffle,
                                batch_size=1)
    print(f'valid dataset size unscaled: {len(valid_dataloader_unscaled.dataset)}')
    
    return test_dataloader_scaled, test_dataloader_unscaled, valid_dataloader_scaled, valid_dataloader_unscaled