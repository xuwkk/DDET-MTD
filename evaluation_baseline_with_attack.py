"""
Run robust MTD algorithm as base line under attack
can be both event trigger and periodic trigger
"""

from copy import deepcopy
from unittest import result
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
from optim.robust_mtd import incomplete_fro, incomplete_l2, x_to_b
from scipy.optimize import minimize

"""
incomplete Robust MTD algorithm
"""

def run_incomplete(v_est_last, case_class, x_max, x_min, col_constraint, k_min):
    
    loss_summary = []
    x_mtd_summary = []
    b_mtd_summary = []
    success_no = 0
    
    v_est_last = v_est_last.numpy().flatten()
    
    case_class = deepcopy(case_class)
    
    # Pre-MTD matrices: does not change during the optimization
    H1 = case_class.H_v_hat(v_est_last)
    H1 = case_class.R_inv_12_@H1
    P1, S1 = case_class.mtd_matrix(H1)   # Projection matrices
    
    # Initialize the robust algorithm
    incomplete_fro_ = incomplete_fro(case_class=case_class, v_est=v_est_last, x_max=x_max, x_min=x_min, col_constraint=col_constraint)

    """
    Multi Run loop
    """
    
    while True:
        
        x0 = x_min + (x_max-x_min)*np.random.rand(len(case_class.x))    # Random initialization
        assert np.all(x0 <= x_max)           
        assert np.all(x0 >= x_min)
    
        # Solve the incomplete_fro problem to find the wart start using SLSQP for l2 problem
        results = minimize(incomplete_fro_.fun_loss, x0, method = "SLSQP", constraints = incomplete_fro_.nonlinear_constraint(), bounds = incomplete_fro_.fun_bound(), options = {'ftol': 1e-5})
        
        x_mtd = results.x
        assert np.all(x_mtd <= x_max + 1e-6)
        assert np.all(x_mtd >= x_min - 1e-6)
        
        # Post MTD matrices
        b_mtd = x_to_b(case_class,x_mtd)
        V, C, Ars, Arc = case_class.H_v_hat_robust(v_est_last)
        C = case_class.R_inv_12_@C
        V = case_class.R_inv_12_@V
        A = Arc
        H = C + V@np.diag(b_mtd)@A   
        P = H@np.linalg.inv(H.T@H)@H.T 
        composite_matrix = np.concatenate([H, H1], axis = -1) 
        rank_com = np.linalg.matrix_rank(composite_matrix)                             
        k = 2*(case_class.no_bus-1) - rank_com 
        
        if k != k_min or results.success == False:
            # Run fro optimization again
            continue
        
        else:
            # if k_min is satisfied, solve the incomplete_l2 optimization problem
            singular_value_non_one_pre = 100                                # Set as a large value to pass the initial break test
            
            for _ in range(20):
                # Maximum iteration number is 10

                # do T-SVD
                U, singular_value, V_transpose = np.linalg.svd(P1@P)        # The SVD can slow the optimization problem
                U_k = U[:,:k_min]                                           # The orhornormal basis of the intersection subspaces
                singular_value_non_one = singular_value[k_min]              # The largest non one singular value
                V_k = V_transpose.T[:,:k_min]                               # The orthonormal basis of the intersection subspaces 
                assert np.max(np.abs(U_k-V_k)) < 1e-6                       # Test the difference between U_k and V_k is small
                
                # instance the incomplete_l2 optimization problem 
                incomplete_l2_ = incomplete_l2(case_class, v_est_last, x_max, x_min, col_constraint, U_k)
                
                x0 = x_mtd                                                  ## use the previous running result as the initial point
                
                results = minimize(incomplete_l2_.fun_loss, x0, method = "SLSQP", constraints = incomplete_l2_.nonlinear_constraint(), bounds = incomplete_l2_.fun_bound(), options = {'ftol': 1e-5})
                
                # analysis
                x_mtd = results.x
                assert np.all(x_mtd <= x_max + 1e-6)
                assert np.all(x_mtd >= x_min - 1e-6)
                
                # Post MTD matrices
                b_mtd = x_to_b(case_class,x_mtd)
                V, C, Ars, Arc = case_class.H_v_hat_robust(v_est_last)
                C = case_class.R_inv_12_@C
                V = case_class.R_inv_12_@V
                A = Arc
                H = C + V@np.diag(b_mtd)@A   
                P = H@np.linalg.inv(H.T@H)@H.T 

                # termination condition
                if np.abs(singular_value_non_one_pre-singular_value_non_one) < 1e-7 and results.success == True:
                    
                    loss_summary.append(singular_value_non_one)     ## record the value of largest non one singular value
                    x_mtd_summary.append(x_mtd)
                    b_mtd_summary.append(b_mtd)
                    
                    # T-SVD
                    U, singular_value, V_transpose = np.linalg.svd(P1@P)         
                    U_k = U[:,:k_min]                                           
                    singular_value_non_one = singular_value[k_min]              
                    V_k = V_transpose.T[:,:k_min]                                             
                    break                                           ## break the incomplete_l2 loop
                
                singular_value_non_one_pre = singular_value_non_one ## update the running result
        
        success_no += 1
        if success_no == 10:       
            break
    x_mtd = x_mtd_summary[np.argmin(loss_summary)]
    b_mtd = b_mtd_summary[np.argmin(loss_summary)]
    return x_mtd, b_mtd, np.min(loss_summary)

if __name__ == "__main__":
    
    """
    settings
    """
    import argparse
    parser = argparse.ArgumentParser()
    
    # train
    parser.add_argument("--is_event_trigger", type = int)
    parser.add_argument("--mtd_type", type = str)
    
    args = parser.parse_args()
    
    mtd_type = args.mtd_type
    is_event_trigger = args.is_event_trigger
    
    total_run = mtd_config['total_run']
    
    if mtd_type == 'robust':
        total_run = mtd_config['total_run']
    elif mtd_type == 'max_rank':
        total_run = mtd_config['total_run'] * 5
    else:
        print('Please input the correct mtd_type')
    
    print('mtd type:', mtd_type, 'event trigger:', is_event_trigger, 'total run', total_run)
    
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

    # Attack list
    ang_no_list = [1,2,3]
    mag_no = 0
    ang_str_list = [0.2,0.3]
    mag_str = 0

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
    x_ratio = {}
    mtd_eff = {}        # The residual of operator at stage one
    mtd_hidden = {}
    cost_no_mtd = {}
    cost_with_mtd = {}
    TP_DDD = {}

    for ang_no in ang_no_list:
        for ang_str in ang_str_list:
            
            print(f'({ang_no},{ang_str})')
            
            """
            Construct the dict keys
            """
            x_ratio[f'({ang_no},{ang_str})'] = []
            mtd_eff[f'({ang_no},{ang_str})'] = []
            mtd_hidden[f'({ang_no},{ang_str})'] = []
            cost_no_mtd[f'({ang_no},{ang_str})'] = []
            cost_with_mtd[f'({ang_no},{ang_str})'] = []
            TP_DDD[f'({ang_no},{ang_str})'] = []
            
            """
            for loop
            """
            for idx_, (idx, input, v_est_pre, v_est_last) in tqdm(enumerate(test_dataloader_unscaled)):  
                """
                idx_: starts from 0
                idx: the actual index number in test_dataloader_unscaled
                """
                
                if idx_ >= int(total_run):
                    # break if the maximum run is reached
                    break
                
                if idx >= len(test_dataloader_unscaled.dataset) - 1:
                    # try another one
                    total_run += 1
                    continue
                
                """
                Convert the data format
                """
                # input: (1,sample_length,feature_size)
                # v_est_pre and v_est_last: (1, no_bus) -> (no_bus,)
                v_est_pre = v_est_pre.flatten()
                v_est_last = v_est_last.flatten()  # The ground truth state
                # true state
                vang_true = np.angle(v_est_last.numpy())
                
                """
                Generate attack
                """
                z_att_noise, v_att_est_last = case_class.gen_fdi_att_dd(z_noise=input, v_est_last=v_est_last, ang_no=ang_no, mag_no=mag_no, ang_str=ang_str, mag_str=mag_str)
                v_att_est_last = torch.from_numpy(v_att_est_last)
                
                if is_event_trigger == 1:
                    
                    """
                    Data-driven detector
                    """
                    # Scale
                    z_att_noise_scale = scaler_(z_att_noise)
                    encoded, decoded, loss_lattent, loss_recons = dd_detector.evaluate(z_att_noise_scale)
                    
                    if loss_recons <= dd_detector.ae_threshold[dd_detector.quantile_idx]:
                        
                        # The attack is not detected
                        TP_DDD[f'({ang_no},{ang_str})'].append(False)
                        
                        """
                        record the cost in case the data-driven detector is not triggered
                        """
                        current_load_idx = test_start_idx + 6 + idx.numpy()   # The index of the current load and pv in the entire load and pv dataset
                        result_no_mtd = case_class.run_opf(load_active = (load_active[current_load_idx]-pv_active_[current_load_idx]), load_reactive = load_reactive[current_load_idx], verbose = False)

                        cost_no_mtd[f'({ang_no},{ang_str})'].append(result_no_mtd['f'])
                        cost_with_mtd[f'({ang_no},{ang_str})'].append(result_no_mtd['f'])
                        x_ratio[f'({ang_no},{ang_str})'].append(np.zeros((case_class.no_brh,)))
                        mtd_eff[f'({ang_no},{ang_str})'].append(0)   # because the MTD is not triggered
                        mtd_hidden[f'({ang_no},{ang_str})'].append(0)
                        continue
                
                    else:
                        TP_DDD[f'({ang_no},{ang_str})'].append(True)

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
                
                """
                Evaluation of the MTD algorithm
                """
                next_load_idx = test_start_idx + 6 + 1 + idx.numpy()   # The index of the next load and pv in the entire load and pv dataset, this is for evaluating the cost
                
                
                
                vang_att = np.angle(v_att_est_last.numpy())
                c_true = (vang_att - vang_true)             # Ground truth angle attack vector with reference bus included
                c_recover_no_ref = np.expand_dims(c_true[case_class.non_ref_index],1)   # No reference bus]

                # Instance the mtd_optim class for evaluation
                mtd_optim_ = mtd_optim(case_class, v_est_last.numpy(), c_recover_no_ref, varrho_square = 0.01)  # Only to instance the class varrho_square is not used
                
                # evaluate the MTD method
                is_converged, x_mtd_change_ratio, cost_no_mtd_, cost_with_mtd_, residual_normal, residual_hid, residual_eff  =  mtd_optim_.mtd_metric_with_attack(b_mtd = b_mtd, 
                                                                                                                                c_actual = c_true, 
                                                                                                                                load_active = load_active[next_load_idx], 
                                                                                                                                load_reactive = load_reactive[next_load_idx], 
                                                                                                                                pv_active = pv_active_[next_load_idx],
                                                                                                                                mode = mtd_config['mode'])
                
                # record the performance
                x_ratio[f'({ang_no},{ang_str})'].append(np.abs(x_mtd - case_class.x)/case_class.x)
                mtd_eff[f'({ang_no},{ang_str})'].append(residual_eff)
                mtd_hidden[f'({ang_no},{ang_str})'].append(residual_hid)
                cost_no_mtd[f'({ang_no},{ang_str})'].append(cost_no_mtd_)
                cost_with_mtd[f'({ang_no},{ang_str})'].append(cost_with_mtd_)
    """
    save
    """
    if is_event_trigger == 0:           
        # save periodic 
        save_metric(address = f'metric/{sys_config["case_name"]}/metric_{mtd_type}_{mtd_config["mode"]}_with_attack_event_{is_event_trigger}.npy', 
                x_ratio = x_ratio,
                mtd_eff = mtd_eff,
                mtd_hidden = mtd_hidden,
                cost_no_mtd = cost_no_mtd,
                cost_with_mtd = cost_with_mtd
                )
    else:
        # save event triggered
        save_metric(address = f'metric/{sys_config["case_name"]}/metric_{mtd_type}_{mtd_config["mode"]}_with_attack_event_{is_event_trigger}.npy', 
                x_ratio = x_ratio,
                mtd_eff = mtd_eff,
                mtd_hidden = mtd_hidden,
                cost_no_mtd = cost_no_mtd,
                cost_with_mtd = cost_with_mtd,
                TP_DDD = TP_DDD
                )