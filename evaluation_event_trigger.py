"""
The full algorithm to run attack detection, extraction, and MTD
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
ang_no_list = [1,2,3]
mag_no = 0
ang_str_list = [0.2,0.3]
mag_str = 0

"""
Metrics
"""
# Attack detection
TP_DDD = {}                   # Record the true positive number of deep learning detector
att_strength = {}             # Record the state attack strength
varrho_summary = {}

# Attack recovery
recover_deviation = {}        # The difference between recovered state phase angle and the ground truth state phase angle
pre_deviation = {}            # Difference by using the previous state phase angle
recovery_ite_no = {}          # The number of iterations in recovery algorithm
recovery_time = []            # The recovery time

# MTD
obj_one = {}                  # Optimal solution of stage one
obj_two = {}                  # Optimal solution of stage two
worst_primal = {}             # Worst attack primal, is compared with the obj_one
worst_dual = {}

fail = {}                     # If the stage one does not have eligible solution

x_ratio_stage_one = {}
x_ratio_stage_two = {}

residual_no_att = []          # BDD residual if there is no attack after MTD

post_mtd_opf_converge = []    # Test if the post mtd opf can converged

mtd_stage_one_time = []       # Overall stage one running time with multi-run considered for each attack
mtd_stage_two_time = []

# MTD detection performance
mtd_stage_one_eff = {}        # The residual of operator at stage one
mtd_stage_two_eff = {}

mtd_stage_one_hidden = {}     # The residual of attacker at stage one
mtd_stage_two_hidden = {}

cost_no_mtd = {}              # The cost withour MTD
cost_with_mtd_one = {}        # The cost with MTD
cost_with_mtd_two = {}


"""
Iterations
"""

for ang_no in ang_no_list:
    # attack number
    for ang_str in ang_str_list:
        # attack strength
        
        print(f'({ang_no},{ang_str})')
        
        """
        Construct the dict keys
        """
        
        # Attack detection
        TP_DDD[f'({ang_no},{ang_str})'] = []                    # Record the true positive number of deep learning detector
        att_strength[f'({ang_no},{ang_str})'] = []              # Record the state attack strength
        varrho_summary[f'({ang_no},{ang_str})'] = []

        # Attack recovery
        recover_deviation[f'({ang_no},{ang_str})'] = []         # The difference between recovered state phase angle and the ground truth state phase angle
        pre_deviation[f'({ang_no},{ang_str})'] = []             # Difference by using the previous state phase angle
        recovery_ite_no[f'({ang_no},{ang_str})'] = []           # The number of iterations in recovery algorithm

        # MTD
        obj_one[f'({ang_no},{ang_str})'] = []                   # Optimal solution of stage one
        obj_two[f'({ang_no},{ang_str})'] = []                   # Optimal solution of stage two
        worst_primal[f'({ang_no},{ang_str})'] = []              # Worst attack primal, is compared with the obj_one
        worst_dual[f'({ang_no},{ang_str})'] = [] 

        fail[f'({ang_no},{ang_str})'] = []
        
        x_ratio_stage_one[f'({ang_no},{ang_str})'] = [] 
        x_ratio_stage_two[f'({ang_no},{ang_str})'] = [] 

        # MTD detection performance
        mtd_stage_one_eff[f'({ang_no},{ang_str})'] = []         # The residual of operator at stage one
        mtd_stage_two_eff[f'({ang_no},{ang_str})'] = [] 

        mtd_stage_one_hidden[f'({ang_no},{ang_str})'] = []      # The residual of attacker at stage one
        mtd_stage_two_hidden[f'({ang_no},{ang_str})'] = [] 

        cost_no_mtd[f'({ang_no},{ang_str})'] = []               # The cost withour MTD
        cost_with_mtd_one[f'({ang_no},{ang_str})'] = []         # The cost with MTD
        cost_with_mtd_two[f'({ang_no},{ang_str})'] = [] 
        
        for idx_, (idx, input, v_est_pre, v_est_last) in tqdm(enumerate(test_dataloader_unscaled)):  
            
            """
            idx_: starts from 0
            idx: the actual index number in test_dataloader_unscaled
            """
            
            if idx_ >= total_run:
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
            
            if loss_recons <= dd_detector.ae_threshold[dd_detector.quantile_idx]:
                # There is no attack, start the next attack
                TP_DDD[f'({ang_no},{ang_str})'].append(False)
                continue
            else:
                TP_DDD[f'({ang_no},{ang_str})'].append(True)
            
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
            recover_deviation[f'({ang_no},{ang_str})'].append(np.linalg.norm(vang_true - vang_recover,2))   # L2 norm
            pre_deviation[f'({ang_no},{ang_str})'].append(np.linalg.norm(vang_true - vang_pre,2))           # L2 norm
            recovery_ite_no[f'({ang_no},{ang_str})'].append(len(loss_recover_summary))
            recovery_time.append(recover_time)
            
            """
            MTD algorithm
            """
            vang_att = np.angle(v_att_est_last.numpy())
            c_true = (vang_att - vang_true)             # Ground truth angle attack vector with reference bus included
            c_recover = (vang_att - vang_recover)       # Recovered angle attack vector with reference bus included
            c_recover_no_ref = np.expand_dims(c_recover[case_class.non_ref_index],1)   # No reference bus]
            
            att_strength[f'({ang_no},{ang_str})'].append(np.linalg.norm(c_recover_no_ref, 2))

            # Instance the mtd optimization class
            mtd_optim_ = mtd_optim(case_class, v_est_last.numpy(), c_recover_no_ref, varrho_square)
            varrho_summary[f'({ang_no},{ang_str})'].append(np.sqrt(mtd_optim_.varrho_square))
        
            # Run MTD stage-one/two
            b_mtd_one_final, b_mtd_two_final, obj_one_final, obj_two_final, obj_worst_primal, obj_worst_dual, c_worst, stage_one_time_, stage_two_time_, is_fail = mtd_optim_.multi_run()
            
            obj_one[f'({ang_no},{ang_str})'].append(obj_one_final)
            obj_two[f'({ang_no},{ang_str})'].append(obj_two_final)
            
            worst_primal[f'({ang_no},{ang_str})'].append(obj_worst_primal)
            worst_dual[f'({ang_no},{ang_str})'].append(obj_worst_dual)

            fail[f'({ang_no},{ang_str})'].append(is_fail)
            
            mtd_stage_one_time.append(stage_one_time_)
            mtd_stage_two_time.append(stage_two_time_)
            
            """
            Evaluate on MTD without hidden (MTD Stage-One result): using ground truth state attack vector
            """
            
            next_load_idx = test_start_idx + 6 + 1 + idx.numpy()   # The index of the next load and pv in the entire load and pv dataset
            
            is_converged, x_mtd_change_ratio, cost_no_mtd_, cost_with_mtd_, residual_normal, residual_hid, residual_eff  =  mtd_optim_.mtd_metric_with_attack(b_mtd = b_mtd_one_final, 
                                                                                                                            c_actual = c_true, 
                                                                                                                            load_active = load_active[next_load_idx], 
                                                                                                                            load_reactive = load_reactive[next_load_idx], 
                                                                                                                            pv_active = pv_active_[next_load_idx],
                                                                                                                            mode = mtd_config['mode'])
            mtd_stage_one_hidden[f'({ang_no},{ang_str})'].append(residual_hid)
            mtd_stage_one_eff[f'({ang_no},{ang_str})'].append(residual_eff)
            x_ratio_stage_one[f'({ang_no},{ang_str})'].append(x_mtd_change_ratio)
            cost_no_mtd[f'({ang_no},{ang_str})'].append(cost_no_mtd_)
            cost_with_mtd_one[f'({ang_no},{ang_str})'].append(cost_with_mtd_)
            
            """
            Evaluate on MTD with hidden (MTD stage-two result)
            """
            is_converged, x_mtd_change_ratio, cost_no_mtd_, cost_with_mtd_, residual_normal_, residual_hid, residual_eff  =  mtd_optim_.mtd_metric_with_attack(b_mtd = b_mtd_two_final, 
                                                                                                                            c_actual = c_true, 
                                                                                                                            load_active = load_active[next_load_idx], 
                                                                                                                            load_reactive = load_reactive[next_load_idx], 
                                                                                                                            pv_active = pv_active_[next_load_idx],
                                                                                                                            mode = mtd_config['mode'])
            mtd_stage_two_hidden[f'({ang_no},{ang_str})'].append(residual_hid)
            mtd_stage_two_eff[f'({ang_no},{ang_str})'].append(residual_eff)
            post_mtd_opf_converge.append(is_converged)
            x_ratio_stage_two[f'({ang_no},{ang_str})'].append(x_mtd_change_ratio)
            cost_with_mtd_two[f'({ang_no},{ang_str})'].append(cost_with_mtd_)
            residual_no_att.append(residual_normal_)

save_metric(address = f'metric/{sys_config["case_name"]}/metric_event_trigger_mode_{mtd_config["mode"]}_{round(np.sqrt(mtd_config["varrho_square"]),5)}_{mtd_config["upper_scale"]}.npy', 
            
            # Attack detection
            TP_DDD = TP_DDD,                   # Record the true positive number of deep learning detector
            att_strength = att_strength,             # Record the state attack strength
            varrho_summary = varrho_summary,

            # Attack recovery
            recover_deviation = recover_deviation,        # The difference between recovered state phase angle and the ground truth state phase angle
            pre_deviation = pre_deviation,            # Difference by using the previous state phase angle
            recovery_ite_no = recovery_ite_no,         # The number of iterations in recovery algorithm
            recovery_time = recovery_time,            # The recovery time

            # MTD
            obj_one = obj_one,                  # Optimal solution of stage one
            obj_two = obj_two,                  # Optimal solution of stage two
            worst_primal = worst_primal,             # Worst attack primal, is compared with the obj_one
            worst_dual = worst_dual,

            fail = fail,                     # If the stage one does not have eligible solution

            x_ratio_stage_one = x_ratio_stage_one,
            x_ratio_stage_two = x_ratio_stage_two,

            residual_no_att = residual_no_att,          # BDD residual if there is no attack after MTD

            post_mtd_opf_converge = post_mtd_opf_converge,    # Test if the post mtd opf can converged

            mtd_stage_one_time = mtd_stage_one_time,       # Overall stage one running time with multi-run considered for each attack
            mtd_stage_two_time = mtd_stage_two_time,

            # MTD detection performance
            mtd_stage_one_eff = mtd_stage_one_eff,        # The residual of operator at stage one
            mtd_stage_two_eff = mtd_stage_two_eff,

            mtd_stage_one_hidden = mtd_stage_one_hidden,     # The residual of attacker at stage one
            mtd_stage_two_hidden = mtd_stage_two_hidden,

            cost_no_mtd = cost_no_mtd,              # The cost without MTD
            cost_with_mtd_one = cost_with_mtd_one,        # The cost with MTD
            cost_with_mtd_two = cost_with_mtd_two

            
            )