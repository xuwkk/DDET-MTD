"""
Test the false positive rejection on deep learning detector using MTD
"""

from utils.load_data import load_case, load_measurement, load_load_pv
from models.model import LSTM_AE
from models.evaluation import Evaluation
import torch
from configs.nn_setting import nn_setting
from configs.config import mtd_config, sys_config, save_metric
import numpy as np
from tqdm import tqdm
from models.dataset import scaler
from optim.optimization import mtd_optim
from torch.utils.data import DataLoader
from models.dataset import rnn_dataset_evl

"""
Preparation
"""

# Load cases, measurement, and load
case_class = load_case()
z_noise_summary, v_est_summary = load_measurement()
load_active, load_reactive, pv_active_, pv_reactive_ = load_load_pv()
feature_size = len(z_noise_summary)
test_start_idx = int(feature_size * (nn_setting['train_prop'] + nn_setting['valid_prop']))  # The start index of test dataset

# Load deep learning detector
lstm_ae = LSTM_AE()
lstm_ae.load_state_dict(torch.load(nn_setting['model_path'], map_location=torch.device(nn_setting['device'])))
dd_detector = Evaluation(case_class=case_class)            # Instance the data-driven detector
scaler_ = scaler()                                         # Instance the scaler class
print(f'Quantile: {dd_detector.quantile[dd_detector.quantile_idx]}')
print(f'Threshold: {dd_detector.ae_threshold[dd_detector.quantile_idx]}')

# The test set dataloader without shuffling
test_dataset_unscaled = rnn_dataset_evl(mode = 'test', istransform=False)
test_dataloader_unscaled = DataLoader(dataset = test_dataset_unscaled, shuffle=False, batch_size=1)

"""
Log
"""
cost_no_mtd = []       # The cost without MTD
cost_with_mtd = []     # The cost with MTD
residual_DDD = []      # The residual of data-driven detector, when MTD is triggered, zero residual is appended
residual_BDD = []      # The BDD residual with MTD included
x_mtd_ratio = []
post_mtd_opf_converge = []
stage_one_time = []
stage_two_time = []
recover_time = []
varrho_summary = []
worst_primal = []
worst_dual = []
obj_one = []
obj_two = []
fail = []
att_str = []


"""
Iteration
"""
posi_trigger = 0

for idx_, (idx, input, v_est_pre, v_est_last) in tqdm(enumerate(test_dataloader_unscaled)):  
    
    if idx_ >= 288*10:
        break
    
    if posi_trigger == 1:
        posi_trigger = 0
        continue
    
    varrho_square = mtd_config['varrho_square']    # The recovery uncertainty
    v_est_pre = v_est_pre.flatten()
    v_est_last = v_est_last.flatten()  # The ground truth state
    
    """
    BDD
    """
    z_noise_last = np.expand_dims(input.numpy()[0,-1],1)
    residual = case_class.bdd_residual(z_noise_last, v_est_last.numpy())
    residual_BDD.append(residual)
    
    """
    Data-driven detector
    """
    # Scale
    input_scale = scaler_(input)
    encoded, decoded, loss_lattent, loss_recons = dd_detector.evaluate(input_scale)
    residual_DDD.append(loss_recons)
    
    current_load_idx = test_start_idx + 6 + idx.numpy()   # The index of the current load and pv in the entire load and pv dataset
    result_no_mtd = case_class.run_opf(load_active = (load_active[current_load_idx]-pv_active_[current_load_idx]), load_reactive = load_reactive[current_load_idx], verbose = False)
    
    cost_no_mtd.append(result_no_mtd['f'])
    cost_with_mtd.append(result_no_mtd['f'])
    
    if loss_recons < dd_detector.ae_threshold[dd_detector.quantile_idx]:
        # Negative detection
        pass
    
    else:
        """
        Attack recover
        """
        # v_recover: the recovered state
        # NOTE: input NOT SCALED
        _, v_recover, _, _, _, _, _, recover_time_ = dd_detector.recover(attack_batch=input, v_pre = v_est_pre, v_last= v_est_last)

        """
        MTD Optimization
        """
        vang_recover = np.angle(v_recover.numpy())  # Recovered angle
        vang_att = np.angle(v_est_last.numpy())     # As if there is an attack on the phase angle
        c_recover = (vang_att - vang_recover)       # Recovered angle attack vector with reference bus included
        c_recover_no_ref = np.expand_dims(c_recover[case_class.non_ref_index],1)   # No reference bus
        
        # Positive Detection
        posi_trigger = 1       # Go to the identification and MTD algorithms
        # Instance the mtd optimization class
        mtd_optim_ = mtd_optim(case_class, v_est_last.numpy(), c_recover_no_ref, varrho_square)
        
        # Run MTD stage-one/two
        b_mtd_one_final, b_mtd_two_final, obj_one_final, obj_two_final, obj_worst_primal, obj_worst_dual, c_worst, stage_one_time_, stage_two_time_, is_fail = mtd_optim_.multi_run()
        
        next_load_idx = test_start_idx + 6 + 1 + idx.numpy()   # The index of the next load and pv in the entire load and pv dataset
        
        # Evaluate the MTD without attack
        is_converged, x_mtd_change_ratio, residual_normal, cost_no_mtd_, cost_mtd_two_ = mtd_optim_.mtd_metric_no_attack(b_mtd = b_mtd_two_final, 
                                                                                                                        load_active = load_active[next_load_idx], 
                                                                                                                        load_reactive = load_reactive[next_load_idx], 
                                                                                                                        pv_active = pv_active_[next_load_idx]
                                                                                                                        )
        
        # Log
        residual_BDD.append(residual_normal)
        residual_DDD.append(0)
        cost_no_mtd.append(cost_no_mtd_)
        cost_with_mtd.append(cost_mtd_two_)
        x_mtd_ratio.append(x_mtd_change_ratio)
        post_mtd_opf_converge.append(is_converged)
        stage_one_time.append(stage_one_time_)
        stage_two_time.append(stage_two_time_)
        recover_time.append(recover_time_)
        varrho_summary.append(np.sqrt(mtd_optim_.varrho_square)) # Check if the uncertainty is reduced
        worst_primal.append(obj_worst_primal)
        worst_dual.append(obj_worst_dual)
        obj_one.append(obj_one_final)
        obj_two.append(obj_two_final)
        fail.append(is_fail)
        att_str.append(np.linalg.norm(c_recover,2))

save_metric(address = f'metric/{sys_config["case_name"]}/metric_fpr_{mtd_config["x_facts_ratio"]}.npy', 
            residual_BDD = residual_BDD,
            residual_DDD = residual_DDD,
            cost_no_mtd = cost_no_mtd,
            cost_with_mtd = cost_with_mtd,
            x_mtd_ratio = np.array(x_mtd_ratio),
            post_mtd_opf_converge = post_mtd_opf_converge,
            stage_one_time = stage_one_time,
            stage_two_time = stage_two_time,
            recover_time = recover_time,
            varrho_summary = varrho_summary,
            worst_primal = worst_primal,
            worst_dual = worst_dual,
            obj_one = obj_one,
            obj_two = obj_two,
            fail = fail,
            att_str = att_str
            )
