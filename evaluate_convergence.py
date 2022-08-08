"""
This file evaluates the convergence of the proposed algorithm
"""

from utils.load_data import load_case, load_measurement, load_load_pv, load_dataset
from models.model import LSTM_AE
from models.evaluation import Evaluation
import torch
from configs.config import sys_config, mtd_config, save_metric
from configs.nn_setting import nn_setting
import numpy as np
from tqdm import tqdm
from models.dataset import scaler
from optim.optimization import mtd_optim

"""
Preparation
"""

total_run = mtd_config['total_run']
print('varrho', np.sqrt(mtd_config['varrho_square']))
print('total_run', total_run)
print('mode', mtd_config['mode'])
print('upper_scaling', mtd_config['upper_scale'])

# Load cases, measurement, and load
case_class = load_case()
z_noise_summary, v_est_summary = load_measurement()
load_active, load_reactive, pv_active_, pv_reactive_ = load_load_pv()
test_dataloader_scaled, test_dataloader_unscaled, valid_dataloader_scaled, valid_dataloader_unscaled = load_dataset()
feature_size = len(z_noise_summary)
test_start_idx = int(feature_size * (nn_setting['train_prop'] + nn_setting['valid_prop']))  # The start index of test dataset

# LSTM-AE detector and identifier
lstm_ae = LSTM_AE()
lstm_ae.load_state_dict(torch.load(nn_setting['model_path'], map_location=torch.device(nn_setting['device'])))
dd_detector = Evaluation(case_class=case_class)  # Instance the data-driven detector
scaler_ = scaler()                                               # Instance the scaler class
print(f'Quantile: {dd_detector.quantile[dd_detector.quantile_idx]}')
print(f'Threshold: {dd_detector.ae_threshold[dd_detector.quantile_idx]}')

# Attack list
ang_no_list = [2]
mag_no = 0
ang_str_list = [0.2]
mag_str = 0

for ang_no in ang_no_list:
    # attack number
    for ang_str in ang_str_list:
        
        for idx_, (idx, input, v_est_pre, v_est_last) in tqdm(enumerate(test_dataloader_unscaled)):  
            
            """
            idx_: starts from 0
            idx: the actual index number in test_dataloader_unscaled
            """
            
            if idx_ >= 1:
                # only run the attack for the first time
                break
            varrho_square = mtd_config["varrho_square"]    # The recovery uncertainty
            
            """
            Convert the data format
            """
            # input: (1,sample_length,feature_size)
            # v_est_pre and v_est_last: (1, no_bus) -> (no_bus,)
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
            
            """
            Attack recover
            """
            # v_recover: the recovered state
            z_recover, v_recover, loss_recover_summary, loss_sparse_real_summary, loss_sparse_imag_summary, loss_v_mag_summary, loss_summary, recover_time = dd_detector.recover(attack_batch=input,   # NOTE: NOT SCALED
                                                                                                                                                        v_pre = v_est_pre,      
                                                                                                                                                        v_last= v_att_est_last, 
                                                                                                                                                        )
            
            # Summary
            vang_recover = np.angle(v_recover.numpy())
            vang_pre = np.angle(v_est_pre.numpy())
            vang_true = np.angle(v_est_last.numpy())
            
            """
            MTD algorithm
            """
            vang_att = np.angle(v_att_est_last.numpy())
            c_true = (vang_att - vang_true)             # Ground truth angle attack vector with reference bus included
            c_recover = (vang_att - vang_recover)       # Recovered angle attack vector with reference bus included
            c_recover_no_ref = np.expand_dims(c_recover[case_class.non_ref_index],1)   # No reference bus]

            # Instance the mtd optimization class
            mtd_optim_ = mtd_optim(case_class, v_est_last.numpy(), c_recover_no_ref, varrho_square)
        
            # Run MTD stage-one/two
            b_mtd_one_final, b_mtd_two_final, obj_one_final, obj_two_final, obj_worst_primal, obj_worst_dual, c_worst, stage_one_time_, stage_two_time_, is_fail = mtd_optim_.multi_run()
            