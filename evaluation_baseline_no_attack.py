"""
assume there is no attack, 
simulate the performance where the MTD is triggered by FPR of the data-driven detector
NOTE: the peoriodic MAX RANK and ROBUST MTDs has same performance on the attack and no attack casa
therefore we only simulate the event-trigger condition
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
from optim.robust_mtd import x_to_b
from evaluation_baseline_with_attack import run_incomplete  # incomplete MTD

if __name__ == "__main__":
    
    """
    settings
    """
    
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--mtd_type", type = str)
    total_run = mtd_config['total_run']
    
    args = parser.parse_args()
    mtd_type = args.mtd_type
    
    if mtd_type == 'robust':
        total_run = mtd_config['total_run']
    elif mtd_type == 'max_rank':
        total_run = mtd_config['total_run'] * 5
    else:
        print('Please input the correct mtd_type')
    
    print('mtd type:', mtd_type, 'total run', total_run)
    
    """
    load everything
    """
    
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
    
    # Constraints
    x_max = case_class.x*(1+mtd_config['x_facts_ratio'])
    x_min = case_class.x*(1-mtd_config['x_facts_ratio'])
    
    # Column constraints
    if sys_config['case_name'] == 'case14':
        degree_one = np.array([8]) - 1 ## buses that are not in any loop
        degree_one_posi = []
        for i in degree_one:
            if i < case_class.ref_index:
                degree_one_posi.append(i)
            elif i > case_class.ref_index:
                degree_one_posi.append(i-1)

        col_constraint = 0.999*np.ones((case_class.no_bus-1,))
        col_constraint[degree_one_posi] = 1
        k_min = 6
    
    # Metrics
    x_ratio = []
    cost_no_mtd = []
    cost_with_mtd = []
    
    posi_trigger = 0 # indicator to detected attack, if equals to 1, then start the next round
    
    for idx_, (idx, input, v_est_pre, v_est_last) in tqdm(enumerate(test_dataloader_unscaled)):  
        """
        idx_: starts from 0
        idx: the actual index number in test_dataloader_unscaled
        """
        
        if idx_ >= int(total_run):
            break
        
        if idx >= len(test_dataloader_unscaled.dataset)-1:
            # try another one
            total_run += 1
            continue
        
        if posi_trigger == 1:
            # we have already store the result and cost (due to MTD trigger) in the previous iteration
            posi_trigger = 0
            continue
        
        v_est_last = v_est_last.flatten()  # The ground truth state
        # true state
        vang_true = np.angle(v_est_last.numpy())
        # Scale
        input_scale = scaler_(input)
        encoded, decoded, loss_lattent, loss_recons = dd_detector.evaluate(input_scale) # detect on the normal meausurement
        
        # normal operation without MTD
        current_load_idx = test_start_idx + 6 + idx.numpy()   # The index of the current load and pv in the entire load and pv dataset
        result_no_mtd = case_class.run_opf(load_active = (load_active[current_load_idx]-pv_active_[current_load_idx]), load_reactive = load_reactive[current_load_idx], verbose = False)
        cost_no_mtd.append(result_no_mtd['f'])
        cost_with_mtd.append(result_no_mtd['f'])
        x_ratio.append(np.zeros((case_class.no_brh,)))
        
        if loss_recons < dd_detector.ae_threshold[dd_detector.quantile_idx]:
            # Negative detection
            pass
        
        else:
            # Positive detection
            posi_trigger = 1  # do not run the next iteration
            # trigger random MTD
            next_load_idx = test_start_idx + 6 + 1 + idx.numpy()   # The index of the next load and pv in the entire load and pv dataset, this is for evaluating the cost
            vang_att = np.angle(v_est_last.numpy())     # As if there is an attack on the phase angle
            c_true = (vang_att - vang_true)             # Ground truth angle attack vector with reference bus included; this is actually zero.
            c_recover_no_ref = np.expand_dims(c_true[case_class.non_ref_index],1)   # No reference bus

            
            """
            MTD algorithm
            """
            if mtd_type == 'robust':
                x_mtd, b_mtd, loss = run_incomplete(v_est_last=v_est_last, 
                                                        case_class = case_class, 
                                                        x_max = x_max, x_min = x_min, 
                                                        col_constraint=col_constraint, 
                                                        k_min = k_min)
                
            elif mtd_type == 'max_rank':
                x_mtd = case_class.random_mtd(x_max_ratio = mtd_config['x_facts_ratio'])
                b_mtd = x_to_b(case_class, x_mtd)
                
            else:
                print("Run MTD algorithm")
                    
            # Instance the mtd_optim class for evaluation
            mtd_optim_ = mtd_optim(case_class, v_est_last.numpy(), c_recover_no_ref, varrho_square = 0.01)  # Only to instance the class c_recover_no_ref and varrho_square is not used
            
            # Evaluate the MTD without attack
            is_converged, x_mtd_change_ratio, residual_normal, cost_no_mtd_, cost_mtd_two_ = mtd_optim_.mtd_metric_no_attack(b_mtd = b_mtd, 
                                                                                                                            load_active = load_active[next_load_idx], 
                                                                                                                            load_reactive = load_reactive[next_load_idx], 
                                                                                                                            pv_active = pv_active_[next_load_idx]
                                                                                                                            )
            
            # log
            cost_no_mtd.append(cost_no_mtd_)
            cost_with_mtd.append(cost_mtd_two_)
            x_ratio.append(np.abs(x_mtd - case_class.x)/case_class.x)
            
    save_metric(address = f'metric/{sys_config["case_name"]}/metric_{mtd_type}_{mtd_config["mode"]}_no_attack.npy', 
                x_ratio = x_ratio,
                cost_no_mtd = cost_no_mtd,
                cost_with_mtd = cost_with_mtd
                )

            