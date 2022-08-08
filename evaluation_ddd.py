"""
Test the performance of attack detection and identification algorithms
"""

from utils.load_data import load_case, load_measurement, load_load_pv, load_dataset
from models.model import LSTM_AE
from models.evaluation import Evaluation
import torch
from configs.nn_setting import nn_setting
from configs.config import sys_config, save_metric
import numpy as np
from tqdm import tqdm
from models.dataset import scaler

# Load cases, measurement, and load
case_class = load_case()
z_noise_summary, v_est_summary = load_measurement()
load_active, load_reactive, pv_active_, pv_reactive_ = load_load_pv()
test_dataloader_scaled, test_dataloader_unscaled, valid_dataloader_scaled, valid_dataloader_unscaled = load_dataset()

lstm_ae = LSTM_AE()
lstm_ae.load_state_dict(torch.load(nn_setting['model_path'], map_location=torch.device(nn_setting['device'])))
dd_detector = Evaluation(case_class=case_class)  # Instance the data-driven detector
scaler_ = scaler()                               # Instance the scaler class
print(f'Quantile: {dd_detector.quantile[dd_detector.quantile_idx]}')
print(f'Threshold: {dd_detector.ae_threshold[dd_detector.quantile_idx]}')

"""
Metrics
"""
# Attack list
ang_no_list = [1,2,3]
mag_no = 0
ang_str_list = [0.2,0.3]
mag_str = 0

# Attack detection
TP_DDD = {}                   # Record the true positive number of deep learning detector
FP_DDD = []

# Attack recovery
att_strength = {}
recover_deviation = {}        # The difference between recovered state phase angle and the ground truth state phase angle
recovery_ite_no = {}          # The number of iterations in recovery algorithm
recovery_time = []            # The recovery time
residual_bdd = {}         # The bdd residual of the recovered measurement
residual_ddd = {}

"""
FPR of DDD
"""
print('FPR of DDD.')

for idx_, (idx, input, v_est_pre, v_est_last) in tqdm(enumerate(test_dataloader_unscaled)):  
    """
    idx_: starts from 0
    idx: the actual index number in test_dataloader_unscaled
    """
    # Scale
    z_att_noise_scale = scaler_(input)
    encoded, decoded, loss_lattent, loss_recons = dd_detector.evaluate(z_att_noise_scale)
    
    if loss_recons <= dd_detector.ae_threshold[dd_detector.quantile_idx]:
        FP_DDD.append(False)
    else:
        FP_DDD.append(True)
    
"""
TPR of DDD
"""
print('TPR of DDD.')

for ang_no in ang_no_list:
    for ang_str in ang_str_list:
        print(f'({ang_no},{ang_str})')
        """
        Construct the dict keys
        """
        TP_DDD[f'({ang_no},{ang_str})'] = []
        att_strength[f'({ang_no},{ang_str})'] = []
        
        for idx_, (idx, input, v_est_pre, v_est_last) in tqdm(enumerate(test_dataloader_unscaled)):  
            """
            idx_: starts from 0
            idx: the actual index number in test_dataloader_unscaled
            """
            
            v_est_pre = v_est_pre.flatten()
            v_est_last = v_est_last.flatten()  # The ground truth state

            """
            Generate attack
            """
            z_att_noise, v_att_est_last = case_class.gen_fdi_att_dd(z_noise=input, v_est_last=v_est_last, ang_no=ang_no, mag_no=mag_no, ang_str=ang_str, mag_str=mag_str)
            v_att_est_last = torch.from_numpy(v_att_est_last)

            """
            Data-driven detector
            """
            # Scale
            z_att_noise_scale = scaler_(z_att_noise)
            encoded, decoded, loss_lattent, loss_recons = dd_detector.evaluate(z_att_noise_scale)
            
            if loss_recons <= dd_detector.ae_threshold[dd_detector.quantile_idx]:
                TP_DDD[f'({ang_no},{ang_str})'].append(False)
            else:
                TP_DDD[f'({ang_no},{ang_str})'].append(True)

            # Attack strength
            vang_true = np.angle(v_est_last.numpy())
            vang_att = np.angle(v_att_est_last.numpy())
            c_true = (vang_att - vang_true)
            att_strength[f'({ang_no},{ang_str})'].append(np.linalg.norm(c_true,2))
            
"""
Unceratainty of attack identification
Random generate attack vector and do attack detection and identification
"""

print('Identification Uncertainry.')

recover_deviation = {}   # The difference between recovered state phase angle and the ground truth state phase angle
pre_deviation = {}       # Difference by using the previous state phase angle
ite_summary = {}         # The number of iterations in recovery
recover_time = {}        # Summary of recovery time

# Attack and recovery
for ang_no in ang_no_list:
    for ang_str in ang_str_list:
        recover_deviation[f'{ang_no,ang_str}'] = []
        pre_deviation[f'{ang_no,ang_str}'] = []
        ite_summary[f'{ang_no,ang_str}'] = []
        recover_time[f'{ang_no,ang_str}'] = []
        residual_bdd[f'{ang_no,ang_str}'] = []
        residual_ddd[f'{ang_no,ang_str}'] = []
        
        for idx_, (idx, input, v_est_pre, v_est_last) in tqdm(enumerate(test_dataloader_unscaled)):  
            
            if idx_ >= 200:
                break
            
            # The dataloader is shuffled
            # idx_: the index of sample
            # idx: the actual index in the test dataset
            # input: (1,sample_length,feature_size)
            # v_est_pre and v_est_last: (1, no_bus) -> (no_bus,)
            
            # Convert format
            v_est_pre = v_est_pre.flatten()
            v_est_last = v_est_last.flatten()  # The ground truth state

            # Attack
            z_att_noise, v_att_est_last = case_class.gen_fdi_att_dd(z_noise=input, v_est_last=v_est_last, ang_no=ang_no, mag_no=mag_no, ang_str=ang_str, mag_str=mag_str)

            v_att_est_last = torch.from_numpy(v_att_est_last)

            # Scale
            z_att_noise_scale = scaler_(z_att_noise)
            encoded, decoded, loss_lattent, loss_recons = dd_detector.evaluate(z_att_noise_scale)
            
            if loss_recons <= dd_detector.ae_threshold[dd_detector.quantile_idx]:
                # There is no attack
                continue
            else:
                pass
            
            # Recovery
            # v_recover: the recovered state
            z_recover, v_recover, loss_recover_summary, loss_sparse_real_summary, loss_sparse_imag_summary, loss_v_mag_summary, loss_summary, recover_time_ = dd_detector.recover(attack_batch=input,   # NOTE: NOT SCALED
                                                                                                                                                        v_pre = v_est_pre,      
                                                                                                                                                        v_last= v_att_est_last
                                                                                                                                                        )
            
            residual_ddd[f'{ang_no,ang_str}'].append(loss_recover_summary[-1])
            
            # Test SE on the recovered attack
            residual_recover_ = case_class.bdd_residual(z_noise=z_recover.numpy(), v_est = v_recover.numpy())
            residual_bdd[f'{ang_no,ang_str}'].append(residual_recover_)
            
            vang_recover = np.angle(v_recover.numpy())
            vang_pre = np.angle(v_est_pre.numpy())
            vang_true = np.angle(v_est_last.numpy())
            recover_deviation[f'{ang_no,ang_str}'].append(np.linalg.norm(vang_true - vang_recover,2))   # L2 norm
            pre_deviation[f'{ang_no,ang_str}'].append(np.linalg.norm(vang_true - vang_pre,2))
            
            ite_summary[f'{ang_no,ang_str}'].append(len(loss_recover_summary))
            recover_time[f'{ang_no,ang_str}'].append(recover_time_)

save_metric(address = f'metric/{sys_config["case_name"]}/metric_ddd_{nn_setting["recover_lr"]}_{nn_setting["beta_real"]}_{nn_setting["beta_imag"]}_{nn_setting["max_step_size"]}_{nn_setting["min_step_size"]}.npy', 
            # Data-driven Detector
            TP_DDD = TP_DDD,
            att_strength = att_strength,
            FP_DDD = FP_DDD,
            recover_deviation = recover_deviation,
            pre_deviation = pre_deviation,
            ite_summary = ite_summary,
            recover_time = recover_time,
            residual_bdd = residual_bdd,
            residual_ddd = residual_ddd
            )