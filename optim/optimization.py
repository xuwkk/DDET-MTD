"""
The optimization problem for stage one and stage two and multi-run and metrix evaluation
In general:
1. The solver may fail to give a solution, for bad start point b?: use try
2. Multi run should be applied: change the start susceptance can result in different solution
3. Numerical errors might occur in the solver, and small floating number to avoid
"""

from ast import Pass
import cvxpy as cp
import numpy as np
from time import time
from copy import deepcopy
from utils.load_data import load_case
from configs.config import mtd_config, sys_config

def b_to_x(case_class, b, x_facts_ratio):

    """
    Given the b, calculate x
    idealy, there is only one feasible solution of x, in case there are multiple solutions, just choose one of them
    """

    x = deepcopy(case_class.x)
    #print(x)
    #print(b)
    r = deepcopy(case_class.r)

    x_min = (1-x_facts_ratio)*x
    x_max = (1+x_facts_ratio)*x

    x_new = np.zeros(len(x),)

    x_new_1 = (-1+np.sqrt(1-4*b**2*r**2 + 1e-5))/(2*b)   # Add small number to avoiud numerical error
    x_new_2 = (-1-np.sqrt(1-4*b**2*r**2 + 1e-5))/(2*b)

    for i in range(len(x)):
        
        if x_new_1[i] >= x_min[i]-1e-4 and x_new_1[i] <= x_max[i]+1e-4:
            x_new[i] = x_new_1[i]
        else:
            x_new[i] = x_new_2[i]
    
    #print(np.abs(x_new - x)/x)
    assert np.all(np.abs(x_new - x)/x <= x_facts_ratio+1e-3)
    
    return x_new

def x_to_b_range(case_class, x_facts_ratio):
    
    """
    From the changing ratio on x to changing ratio on b used in optimization problem
    x_facts_ratio: the maximum changing ratio on reactance
    """

    x = deepcopy(case_class.x)
    r = deepcopy(case_class.r)

    x_min = (1-x_facts_ratio)*x
    x_max = (1+x_facts_ratio)*x

    b_max = np.zeros(len(x),)
    b_min = np.zeros(len(x),)

    for i in range(len(x_min)):
        if r[i] <= x_max[i] and r[i] >= x_min[i]:
            # In the middle
            b_min[i] = -r[i]/(r[i]**2+r[i]**2)
            b_max[i] = np.max([-x_max[i]/(x_max[i]**2+r[i]**2), -x_min[i]/(x_min[i]**2+r[i]**2)])
        else:
            b_min[i] = np.min([-x_max[i]/(x_max[i]**2+r[i]**2), -x_min[i]/(x_min[i]**2+r[i]**2)])
            b_max[i] = np.max([-x_max[i]/(x_max[i]**2+r[i]**2), -x_min[i]/(x_min[i]**2+r[i]**2)])

    assert np.all(b_min <= b_max + 1e-6)

    return b_max, b_min

class mtd_optim:
    """
    Implement the MTD stage-one and stage-two optimization algorithm
    For each angle attack recovery instance, generate one class instance
    """
    
    def __init__(self, case_class, v_est, c_recover_no_ref, varrho_square):
        """
        Prepare the matrices and constants for MTD convex optimization
        case_class: the instance class case
        v_est: the estimated state before mtd
        x_facts_ratio: the perturbation range of reactance
        bmin/bmax: the range of the susceptance
        c_recover_no_ref: the recovered phase angle attack vector by DDD
        varrho_square: the uncertainty of the DDD recovery (in |.|^2)
        
        # Hyperparameters: is defined in the sys_configs
        
        upper_scale: the increasing threshold of the stage one MTD to encure numerical stability in stage two
        tol_one: the stopping tolerance of stage one problem
        tol_two: the stopping tolerance of stage two problem
        max_ite: the maximum iteration in stage-one convex optimization
        multi_run_no: the multirun number for each attack instance
        """
        
        """
        Define constants
        """
        
        # H_v_hat: approximate the attack vector
        H1 = case_class.H_v_hat(v_est=v_est)
        # H0: approximate the MTD effectiveness
        v_est_ref = v_est[case_class.ref_index]
        V, C, _, Arc = case_class.H_v_flat(vang_ref = np.angle(v_est_ref), vmag_ref = np.abs(v_est_ref))  # Arc: with tap ratio considered
        # Normalization
        self.H1 = case_class.R_inv_12_@H1
        self.C = case_class.R_inv_12_@C
        self.V = case_class.R_inv_12_@V
        self.A = Arc
        # Suceptance range
        x_facts_ratio = mtd_config['x_facts_ratio']
        self.bmax, self.bmin = x_to_b_range(case_class, x_facts_ratio=x_facts_ratio)  # Convert the reactance range into susceptance range
        
        """
        Define parameters
        """
        
        self.no_brh = case_class.no_brh
        self.no_bus = case_class.no_bus - 1 # Remove the reference bus
        self.no_mea = case_class.no_mea
        self.case_class = case_class
        self.bori = case_class.b            # The original susceptance: used in hidden MTD
        self.x_facts_ratio = x_facts_ratio
        
        self.c_recover_no_ref = c_recover_no_ref
        self.varrho_square = varrho_square
        self.v_est = v_est
        
        self.upper_scale = mtd_config['upper_scale']                         # Increase the bdd threshold a little bit
        self.threshold = case_class.bdd_threshold_pf                         # Power flow measurement threshold
        self.threshold_one = case_class.bdd_threshold_pf * self.upper_scale  # Increased threshold in stage-one/two problem
        self.tol_one = mtd_config['tol_one']                                 # Stopping tolerance in stage one
        self.tol_two = mtd_config['tol_two']                                 # Stopping tolerance in stage two
        self.max_ite = mtd_config['max_ite']                                 # Maximum iteration number in single convex optimization
        self.multi_run_no = mtd_config['multi_run_no']                       # Multi-run number in stage one
        self.verbose = mtd_config['verbose']                                 # Show optimization details or not
        self.is_worst = mtd_config['is_worst']                               # Calculate the worst attack or not
        
        """
        Update the uncertainty varrho_square
        """
        self.att_str = np.linalg.norm(c_recover_no_ref, 2)
        if self.att_str <= np.sqrt(varrho_square)*1.2:
            self.varrho_square = (self.att_str*0.6)**2
        
    def random_b0(self):
        """
        Randomly generate the initial susceptance used in stage-one
        """  
        assert np.all(self.bmin <= self.bmax + 1e-3)
        # Initial point in the update
        b0 = self.bmin + (self.bmax - self.bmin)*np.random.rand(self.case_class.no_brh)
        # Test the legit data
        assert np.all(b0<=self.bmax)
        assert np.all(b0>=self.bmin)
        
        return b0
    
    """
    Worst Attack
    """
    
    def worst_attack(self, b_mtd):

        """
        Find the vulnerable attack for a given MTD strategy
        b_mtd: the susceptance after MTD, NEGATIVE
        """
        
        # Calculate the MTD matrices
        H0 = self.C-self.V@np.diag(b_mtd)@self.A
        S0 = np.eye(self.no_brh) - H0@np.linalg.inv(H0.T@H0)@H0.T    # A constant as the b_mtd is given
        
        omega = cp.Variable()          # Objective variable
        nu = cp.Variable()             # Dual Variable
        
        objective = cp.Maximize(omega) # Objective function: maximize the dual function

        constraints = []
        constraints.append(nu>=0)  
        constraints.append(cp.bmat(
            [
                [nu*(self.c_recover_no_ref.T@self.c_recover_no_ref - self.varrho_square) - omega,    nu*self.c_recover_no_ref.T],
                [(nu*self.c_recover_no_ref.T).T,                                                     self.H1.T@S0@self.H1+nu*np.eye(self.no_bus)]
            ]
        )>>0)
        
        prob = cp.Problem(objective, constraints)
        result = prob.solve(verbose = False, solver = 'MOSEK')
        
        # Find the primal optimum
        c_worst = nu.value*np.linalg.pinv(self.H1.T@S0@self.H1+nu.value*np.eye(self.no_bus))@self.c_recover_no_ref
        primal_optimum = (S0@self.H1@c_worst).T@(S0@self.H1@c_worst)
        
        return float(omega.value), float(primal_optimum[0,0]), c_worst  # Dual optimum, Primal optimum, worst attack in the range
    
    """
    Stage One
    """
    
    def stage_one(self):
        
        # Preparation
        obj_pre = 0   # Maximize
        ite = 0  
        c = self.c_recover_no_ref
        b0 = self.random_b0()    # Random generage initial susceptance to update
        
        omega_sequence = []
        
        while True:
            
            ite += 1
            # Construct the problem 
            b = cp.Variable(self.no_brh)     # Primal Variable
            nu = cp.Variable()               # Dual Variable
            omega = cp.Variable()            # Objective variable
            
            objective = cp.Maximize(omega)   # Objective function
            
            constraints = []
            constraints.append(cp.diag(b) - cp.diag(self.bmin) >> 0)   # Primal lower bound
            constraints.append(cp.diag(self.bmax) - cp.diag(b) >> 0)   # Primal upper bound
            constraints.append(nu>=0)                                  # Dual feasible
            constraints.append(cp.bmat(
                [
                    [nu*(c.T@c - self.varrho_square) - omega,    nu*c.T,                                             np.zeros((1,self.no_bus))],
                    [(nu*c.T).T,                                 nu*np.eye(self.no_bus) + self.H1.T@self.H1,         self.H1.T@(self.C - self.V@cp.diag(b)@self.A)],
                    [(np.zeros((1,self.no_bus))).T,              (self.H1.T@(self.C - self.V@cp.diag(b)@self.A)).T,  (self.V@cp.diag(b0)@self.A).T@(self.C+self.V@cp.diag(b)@self.A) + (self.C+self.V@cp.diag(b)@self.A).T@(self.V@cp.diag(b0)@self.A) - (self.V@cp.diag(b0)@self.A).T@(self.V@cp.diag(b0)@self.A)]
                ]  
                    )>> 0)
            
            prob = cp.Problem(objective, constraints)
            
            try:
                # In case the problem cannot be solved
                result = prob.solve(verbose = False, solver = 'MOSEK')
            except:
                # Find a new starting point and continue
                b0 = self.random_b0() 
                continue
            
            if self.verbose:
                print(f'{omega.value}')
                omega_sequence.append(omega.value)
                
            # Terminate condition
            if np.abs(obj_pre - omega.value) <= self.tol_one or omega.value >= self.threshold_one or ite >= self.max_ite:
                break
            
            # Update b0 and optimum
            b0 = b.value
            obj_pre = omega.value
        
        if self.verbose:
            np.save(f'metric/{sys_config["case_name"]}/omega_stage_one_{round(np.sqrt(mtd_config["varrho_square"]),5)}.npy', omega_sequence)
        
        return b.value.flatten(), float(omega.value)  # susceptance. optimum

    """
    Stage Two
    """
    def stage_two(self, b_mtd_stage_one, threshold_new):
        
        """
        Stage-two mtd convex optimization
        b_mtd_stage_one: we use the mtd from stage one as the warm start of stage two
        threshold_new: the calculated threshold from stage one, either the bdd_threshold_pf or a smaller value (when the stage one problem cannot reach the predefined threshold)
        """

        # Preparation
        bori = np.expand_dims(self.bori,1)    # The original susceptance
        H_hid = self.case_class.hidden_matrix(v_est=self.v_est)
        b0 = b_mtd_stage_one                  # Start from the solution of stage-one problem
        c = self.c_recover_no_ref             # Identified angle attack vector
        
        # Update loop
        obj_pre = 1e5   # Minimize
        ite = 0

        omega_sequence = []
        
        while True:
            
            ite += 1
            # Construct the problem 
            b = cp.Variable((self.no_brh,1))      # Primal Variable
            nu = cp.Variable()                    # Dual Variable
            omega = cp.Variable((1,1))            # Objective variable

            objective = cp.Minimize(omega)   # Objective function

            constraints = []
            constraints.append(cp.diag(b) - cp.diag(self.bmin) >> 0)   # Primal lower bound
            constraints.append(cp.diag(self.bmax) - cp.diag(b) >> 0)   # Primal upper bound
            constraints.append(nu>=0)                                  # Dual feasible
            
            # Objective constraints
            constraints.append(cp.bmat(
                [
                    [omega,                    (b-bori).T@H_hid.T],
                    [((b-bori).T@H_hid.T).T,   np.eye(self.no_mea)]
                ]
                    ) >> 0)

            # Constraint constraints
            constraints.append(cp.bmat(
                [
                    [nu*(c.T@c - self.varrho_square) - threshold_new,         nu*c.T,                                             np.zeros((1,self.no_bus))],
                    [(nu*c.T).T,                                              nu*np.eye(self.no_bus) + self.H1.T@self.H1,         self.H1.T@(self.C - self.V@cp.diag(b)@self.A)],
                    [(np.zeros((1,self.no_bus))).T,                           (self.H1.T@(self.C - self.V@cp.diag(b)@self.A)).T,  (self.V@cp.diag(b0)@self.A).T@(self.C+self.V@cp.diag(b)@self.A) + (self.C+self.V@cp.diag(b)@self.A).T@(self.V@cp.diag(b0)@self.A) - (self.V@cp.diag(b0)@self.A).T@(self.V@cp.diag(b0)@self.A)]
                ]  
                    )>> 0)

            prob = cp.Problem(objective, constraints)

            try:
                result = prob.solve(verbose = False, solver = 'MOSEK')
            except:
                # If there is no solution
                return b0.flatten(), float(obj_pre)

            if self.verbose:
                print(f'{omega.value[0,0]}')
                omega_sequence.append(omega.value)
                
            # Terminate condition
            if np.abs(obj_pre - omega.value) <= self.tol_two or ite >= self.max_ite:
                break
            
            #if ite >= 20 and omega.value >= 1000:
            #    break
            
            # Update 
            b0 = b.value
            obj_pre = omega.value[0,0]
            
        if self.verbose:
            np.save(f'metric/{sys_config["case_name"]}/omega_stage_two_{round(np.sqrt(mtd_config["varrho_square"]),5)}.npy', omega_sequence)

        return b.value.flatten(), float(omega.value)
    
    """
    The multi-run of the convex stage-one and stage-two problem
    """
    def multi_run(self):
        
        # Log for stage one
        b_candidate_one = []
        obj_candidate_one = []
        
        """
        MTD stage-one
        """
        start_time = time()
        for ite in range(self.multi_run_no):
            b_mtd_one, obj_one = self.stage_one()
            b_candidate_one.append(b_mtd_one)
            obj_candidate_one.append(obj_one)
            
        end_time = time()
        stage_one_time = end_time - start_time
        
        b_candidate_one = np.array(b_candidate_one)     # Convert to numpy array
        obj_candidate_one = np.array(obj_candidate_one)
        
        """
        MTD stage-two
        """ 
        
        if np.max(obj_candidate_one) < 0:
            """
            Stage one fails
            """
            # If stage one does not have eligibale solution, do not run stage two
            b_mtd_one_final = b_candidate_one[np.argmax(obj_candidate_one)]
            b_mtd_two_final = b_candidate_one[np.argmax(obj_candidate_one)]
            obj_one_final = 0
            obj_two_final = 1000
            stage_two_time = 0
            
            is_fail = 1
            
            if self.is_worst:
                # Calculate the worst attack on stage-two result
                obj_worst_dual, obj_worst_primal, c_worst = self.worst_attack(b_mtd_one_final)
                return b_mtd_one_final, b_mtd_two_final, obj_one_final, obj_two_final, obj_worst_primal, obj_worst_dual, c_worst, stage_one_time, stage_two_time, is_fail
            else:
                # The time includes the multi-run
                return b_mtd_one_final, b_mtd_two_final, obj_one_final, obj_two_final, stage_one_time, stage_two_time, is_fail
        
        else:
            """
            Stage one success
            """
            
            is_fail = 0
            
            # Find the eligible susceptance
            idx = np.where(obj_candidate_one >= self.threshold_one)[0]  # Index that is larger than the stage-one threshold
            
            # b_candidate_one_: the susceptance from stage one that is eligible in stage two
            # obj_candidate_one_: the objective from stage one that is eligible in stage two
            
            if len(idx) == 0:
                # If there is no eligible mtd, find the the one with the largest residual
                b_candidate_one_ = b_candidate_one[np.argmax(obj_candidate_one)]
                obj_candidate_one_ = np.max(obj_candidate_one)
                threshold_new = np.max(obj_candidate_one)/(self.upper_scale)     # Set the threshold in stage two
                #threshold_new = np.max(obj_candidate_one)/1.01
                    
            else:
                # There is eligible mtd with residual higher than the threshold
                b_candidate_one_ = b_candidate_one[idx]
                obj_candidate_one_ = obj_candidate_one[idx]
                threshold_new = self.threshold                                 # The predefined bdd threshold
            
            if b_candidate_one_.ndim == 1:
                # If there is only one candidate, add one dimension
                b_candidate_one_ = np.expand_dims(b_candidate_one_, 0)         
                obj_candidate_one_ = np.expand_dims(obj_candidate_one_, 0)
    
            # Log
            b_candidate_two = []
            obj_candidate_two = []
            
            start_time = time()
            for ite in range(len(b_candidate_one_)):
                # Do stage-two for each of the candidate b from stage-one
                b_mtd_two, obj_two = self.stage_two(b_candidate_one_[ite], threshold_new)  # threshold_new is either pf threshold or a smaller threshold
                b_candidate_two.append(b_mtd_two)
                obj_candidate_two.append(obj_two)
            end_time = time()
            stage_two_time = end_time - start_time
            
            # Find the stage two solution
            b_candidate_two = np.array(b_candidate_two)  
            obj_two_final = np.min(obj_candidate_two)
            b_mtd_two_final = b_candidate_two[np.argmin(obj_candidate_two)].flatten() 
            
            # Find the corresponding stage one solution
            #print(f'candidate b: {obj_candidate_one_}')
            b_mtd_one_final = b_candidate_one_[np.argmin(obj_candidate_two)]    # Use the b_mtd_one that results in b_mtd_two_fi
            obj_one_final = obj_candidate_one_[np.argmin(obj_candidate_two)]
            
            if self.is_worst:
                # Calculate the worst attack on stage-two result
                obj_worst_dual, obj_worst_primal, c_worst = self.worst_attack(b_mtd_one_final)
                return b_mtd_one_final, b_mtd_two_final, obj_one_final, obj_two_final, obj_worst_primal, obj_worst_dual, c_worst, stage_one_time, stage_two_time, is_fail
                
            else:
                # The time includes the multi-run
                return b_mtd_one_final, b_mtd_two_final, obj_one_final, obj_two_final, stage_one_time, stage_two_time, is_fail
    
    """
    Evaluation metrics
    """
    
    def mtd_metric_with_attack(self, b_mtd, c_actual, load_active, load_reactive, pv_active, mode):
        """
        Given the mtd solution, return the effectiveness and hiddenness of the MTD
        b_mtd: the solution of the stage-one/two MTD algoriehm
        c_actual: the ground truth angle attack vector
        load_active/reactive/pv: the NEXT INSTANCE load (the pv_active should match the dimension of load_active)
        mode-0: the attacker uses the GROUND TRUTH state (after MTD) to generate attack
        mode-1: the attacker uses the ESTIMATED state to generate attack
        """
        
        # Run opf on the old model
        result = self.case_class.run_opf(load_active = (load_active-pv_active), load_reactive = load_reactive, verbose = False)    
        
        # Reactance change ratio
        x_mtd = b_to_x(case_class=self.case_class, b = b_mtd, x_facts_ratio=self.x_facts_ratio)   # Convert susceptance to reactance
        x_mtd_change_ratio = (x_mtd - self.case_class.x)/self.case_class.x
        assert np.all(np.abs(x_mtd_change_ratio) <= self.x_facts_ratio+1e-3)
        
        # Generate new case
        case_class_new = load_case()
        case_class_new.update_reactance(x_new = x_mtd)   # Update the reactance by the MTD results
        
        # Generage new measurement based on the new model
        result_new = case_class_new.run_opf(load_active = (load_active-pv_active), load_reactive = load_reactive, verbose = False) 
        z_new, z_noise_new, vang_ref_new, vmag_ref_new = case_class_new.construct_mea(result_new)
        
        """
        BDD: System operator on the new measurement based on the new model (because SO knows)
        """
        v_est_ope_no_att = case_class_new.ac_se_pypower(z_noise = z_noise_new, vang_ref = vang_ref_new, vmag_ref=vmag_ref_new, is_honest=True)   # SO state estimation: no attack
        residual_ope_no_att = case_class_new.bdd_residual(z_noise=z_noise_new, v_est = v_est_ope_no_att)                                         # SO residual: no attack
        
        """
        Hiddenness: the attacker uses the old model to test the new measurement
        """
        v_est_attacker = self.case_class.ac_se_pypower(z_noise = z_noise_new, vang_ref = vang_ref_new, vmag_ref=vmag_ref_new, is_honest=True)    # Attacker state estimaion
        residual_attacker = self.case_class.bdd_residual(z_noise = z_noise_new, v_est = v_est_attacker)

        """
        Effectiveness: the SO uses the new model on the attack vector
        the attacker uses the estimated state or the ground truth to construct the attack vector
        NOTE: The attacker can either use the estimated state or the actual state to generate attack
        """
        if mode == 1:
            # Using the attacker's estimated state
            z_noise_new_att = self.case_class.gen_fdi_att_ang(z_noise_new=z_noise_new, v_est_attacker=v_est_attacker, c_actual=c_actual)              # Generate the attack based on the old model (the attack on the angle is unchanged)
        elif mode == 0:
            # Using the ground truth state after MTD
            z_noise_new_att = self.case_class.gen_fdi_att_ang(z_noise_new=z_noise_new, v_est_attacker=v_est_ope_no_att, c_actual=c_actual)
        
        v_est_ope_with_att = case_class_new.ac_se_pypower(z_noise=z_noise_new_att, vang_ref=vang_ref_new, vmag_ref=vmag_ref_new, is_honest=True)      # Operator: Do state estimation based on the new model and the attacked measurement
        residual_ope_with_att = case_class_new.bdd_residual(z_noise=z_noise_new_att, v_est = v_est_ope_with_att)
        
        return result_new['success'], x_mtd_change_ratio, result['f'], result_new['f'], residual_ope_no_att, residual_attacker, residual_ope_with_att
    
    def mtd_metric_no_attack(self, b_mtd, load_active, load_reactive, pv_active):
        """
        Similarly to the with attack case but without attack
        """
        
        # Run opf on the old model (with MTD)
        result = self.case_class.run_opf(load_active = (load_active-pv_active), load_reactive = load_reactive, verbose = False)    
        
        # Reactance change ratio
        x_mtd = b_to_x(case_class=self.case_class, b = b_mtd, x_facts_ratio=self.x_facts_ratio)   # Convert susceptance to reactance
        x_mtd_change_ratio = (x_mtd - self.case_class.x)/self.case_class.x
        assert np.all(np.abs(x_mtd_change_ratio) <= self.x_facts_ratio+1e-3)
        
        # Generate new case
        case_class_new = load_case()
        case_class_new.update_reactance(x_new = x_mtd)   # Update the reactance by the MTD results
        
        # Generage new measurement based on the new model
        result_new = case_class_new.run_opf(load_active = (load_active-pv_active), load_reactive = load_reactive, verbose = False)
        z_new, z_noise_new, vang_ref_new, vmag_ref_new = case_class_new.construct_mea(result_new)
        
        """
        BDD: System operator on the new measurement based on the new model (because SO knows)
        """
        v_est_ope_no_att = case_class_new.ac_se_pypower(z_noise = z_noise_new, vang_ref = vang_ref_new, vmag_ref=vmag_ref_new, is_honest=True) 
        residual_ope_no_att = case_class_new.bdd_residual(z_noise=z_noise_new, v_est = v_est_ope_no_att)
        
        return result_new['success'], x_mtd_change_ratio, residual_ope_no_att, result['f'], result_new['f']
        # if converged, the final mtd change ratio, the residual of post MTD, the cost if there is no MTD, the cost with MTD

# def optim_preparation_stage_one(case_class, v_est):

#     """
#     Prepare the matrices and constants for MTD stage one convex optimization
#     case_class: the instance class case
#     v_est: the estimated state before mtd
#     bmin/bmax: the range of the susceptance
#     """
    
#     # H_v_hat: approximate the attack vector
#     H1 = case_class.H_v_hat(v_est=v_est)
    
#     # H0: approximate the MTD
#     v_est_ref = v_est[case_class.ref_index]
#     V, C, _, Arc = case_class.H_v_flat(vang_ref = np.angle(v_est_ref), vmag_ref = np.abs(v_est_ref))  # Arc: with tap ratio considered
    
#     # vang_flat, vmag_flat = case_class.construct_v_flat(vang_ref = np.angle(v_est_ref), vmag_ref = np.abs(v_est_ref))
#     # v_flat = vmag_flat*np.exp(1j*vang_flat)
#     # dpf_dvang_flat_fun = case_class.H_v_hat(v_est=v_flat)
#     # dpf_dvang_flat_equ = C-V@np.diag(case_class.b)@Arc
#     # print(f'max difference: {np.max(np.abs(dpf_dvang_flat_equ - dpf_dvang_flat_fun))}')
    
#     # Normalization
#     H1 = case_class.R_inv_12_@H1
#     C = case_class.R_inv_12_@C
#     V = case_class.R_inv_12_@V
#     A = Arc
    
#     return H1, C, V, A

# def random_b0(case_class, bmin, bmax):
#     """
#     Randomly generate the susceptance
#     """  
#     assert np.all(bmin <= bmax + 1e-6)
#     # Initial point in the update
#     b0 = bmin + (bmax - bmin)*np.random.rand(case_class.no_brh)
#     # Test the legit data
#     assert np.all(b0<=bmax)
#     assert np.all(b0>=bmin)
#     return b0

# def worst_attack(case_class, v_est, bmax, bmin, c, varrho_square, b_mtd):

#     """
#     Find the vulnerable attack for a given MTD strategy
#     b_mtd: the susceptance after MTD, negative
#     """
    
#     # Calculate the matrices
#     H1, C, V, A= optim_preparation_stage_one(case_class, v_est)
#     no_brh = len(bmin)   # the number of branches
#     no_bus = H1.shape[1] # the number of bus (without reference bus)
    
#     # Calculate the MTD matrix
#     H0 = C-V@np.diag(b_mtd)@A
#     S0 = np.eye(no_brh) - H0@np.linalg.inv(H0.T@H0)@H0.T    # A constant as the b_mtd is given
    
#     omega = cp.Variable()          # Objective variable
#     nu = cp.Variable()             # Dual Variable
    
#     objective = cp.Maximize(omega) # Objective function

#     constraints = []
#     constraints.append(nu>=0)  
#     constraints.append(cp.bmat(
#         [
#             [nu*(c.T@c - varrho_square) - omega,    nu*c.T],
#             [(nu*c.T).T,                            H1.T@S0@H1+nu*np.eye(no_bus)]
#         ]
#     )>>0)
    
#     prob = cp.Problem(objective, constraints)
#     result = prob.solve(verbose = False, solver = 'CVXOPT')
    
#     # Find the primal optimum
#     # print(H1.T@S0@H1+nu*np.eye(no_bus))
#     c_worst = nu.value*np.linalg.pinv(H1.T@S0@H1+nu.value*np.eye(no_bus))@c
#     primal_optimum = (S0@H1@c_worst).T@(S0@H1@c_worst)
    
#     return omega.value, primal_optimum[0,0], c_worst

# def stage_one(case_class, v_est, bmin, bmax, c, varrho_square, threshold, tol, max_ite):

#     """
#     >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#     case_class: the case class instance
#     v_est: the state estimation before MTD
#     bmin: the minimum susceptance, negative
#     bmax: the maximum susceptance, negative
#     c: the recovered attack vector from DDD
#     varrho_square: the uncertainty upper bound for c, with square
#     threshold: the BDD threshold for active power flow measurement
#     tol: the termination threshold for iterative SDPs
#     max_ite: the maximum iteration number for SDPs.

#     >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#     OTHER PARAMETERS:

#     H1: R^-1/2 * H_v_hat   # Constant
#     C: R^-1/2 * C          # Constant
#     V: R^-1/2 * V0         # Constant
#     A: [1/tau] * Ar^cos    # Constant
#     b0: the initial value of b
#     >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#     """
    
#     # Calculate the matrices
#     H1, C, V, A = optim_preparation_stage_one(case_class, v_est)
#     b0 = random_b0(case_class=case_class, bmax=bmax, bmin=bmin)    # Random generate initial susceptance
#     no_brh = len(bmin)     # the number of branches
#     no_bus = H1.shape[1]   # the number of bus (without reference bus)
    
#     # Update loop
#     obj_pre = 0
#     ite = 0

#     while True:
        
#         ite += 1
#         # Construct the problem 
#         b = cp.Variable(no_brh)          # Primal Variable
#         nu = cp.Variable()               # Dual Variable
#         omega = cp.Variable()            # Objective variable
        
#         objective = cp.Maximize(omega)   # Objective function
        
#         constraints = []
#         constraints.append(cp.diag(b) - cp.diag(bmin) >> 0)   # Primal lower bound
#         constraints.append(cp.diag(bmax) - cp.diag(b) >> 0)   # Primal upper bound
#         constraints.append(nu>=0)                             # Dual feasible
#         constraints.append(cp.bmat(
#             [
#                 [nu*(c.T@c - varrho_square) - omega,    nu*c.T,                         np.zeros((1,no_bus))],
#                 [(nu*c.T).T,                            nu*np.eye(no_bus) + H1.T@H1,    H1.T@(C - V@cp.diag(b)@A)],
#                 [(np.zeros((1,no_bus))).T,              (H1.T@(C - V@cp.diag(b)@A)).T,  (V@cp.diag(b0)@A).T@(C+V@cp.diag(b)@A) + (C+V@cp.diag(b)@A).T@(V@cp.diag(b0)@A) - (V@cp.diag(b0)@A).T@(V@cp.diag(b0)@A)]
#             ]  
#                 )>> 0)
        
#         prob = cp.Problem(objective, constraints)
        
#         try:
#             # In case the problem cannot be solved
#             result = prob.solve(verbose = False, solver = 'MOSEK')
#         except:
#             # Find a new starting point and continue
#             # print(f'No solution, restarting...')
#             b0 = random_b0(case_class=case_class, bmax=bmax, bmin=bmin) 
#             continue

#         #print(f'b_opt: {b.value}')
#         #print(f'omega: {omega.value}')
        
#         # Update b0
#         b0 = b.value
        
#         # Terminate condition
#         if np.abs(obj_pre - omega.value) <= tol or omega.value >= threshold or ite == max_ite:
#             break
        
#         obj_pre = omega.value
    
#     return b.value, omega.value
    
# def stage_two(case_class, v_est, bmin, bmax, bori, b0, c, varrho_square, threshold_new, tol, max_ite):
#     """
#     >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#     Stage two problem: considers the MTD hiddenness
#     >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#     case_class: the case class instance
#     v_est: the state estimation before MTD
#     bmin: the minimum susceptance, negative
#     bmax: the maximum susceptance, negative
#     bori: the default susceptance
#     b0: the starting point in update (the result calculated from stage one)
#     c: the recovered attack vector from DDD
#     varrho_square: the uncertainty upper bound for c, with square
#     threshold_new: the threshold of the effectiveness constraints, 
#                     can be a lower one in case the effectiveness is not satisfied.
#     tol: the termination threshold for iterative SDPs
#     max_ite: the maximum iteration number for SDPs

#     >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#     """

#     no_brh = len(bmin)   # the number of branches
#     no_mea = case_class.no_mea
#     no_bus = case_class.no_bus -1   #  Don't count the reference bus

#     bori = np.expand_dims(bori,1)

#     H1, C, V, A = optim_preparation_stage_one(case_class, v_est)
#     # Calculate H_hid = R^{-1/2}@S_v_hat@H_b
#     H_hid = case_class.hidden_matrix(v_est=v_est)
    
#     # Update loop
#     obj_pre = 1e5
#     ite = 0

#     while True:
        
#         ite += 1
#         # Construct the problem 
#         b = cp.Variable((no_brh,1))          # Primal Variable
#         nu = cp.Variable()               # Dual Variable
#         omega = cp.Variable((1,1))            # Objective variable

#         objective = cp.Minimize(omega)   # Objective function

#         constraints = []
#         constraints.append(cp.diag(b) - cp.diag(bmin) >> 0)   # Primal lower bound
#         constraints.append(cp.diag(bmax) - cp.diag(b) >> 0)   # Primal upper bound
#         constraints.append(nu>=0)                             # Dual feasible
        
#         # Objective constraints
#         constraints.append(cp.bmat(
#             [
#                 [omega,                    (b-bori).T@H_hid.T],
#                 [((b-bori).T@H_hid.T).T,   np.eye(no_mea)]
#             ]
#                 ) >> 0)

#         # Constraint constraints
#         constraints.append(cp.bmat(
#             [
#                 [nu*(c.T@c - varrho_square) - threshold_new,    nu*c.T,                         np.zeros((1,no_bus))],
#                 [(nu*c.T).T,                                    nu*np.eye(no_bus) + H1.T@H1,    H1.T@(C - V@cp.diag(b)@A)],
#                 [(np.zeros((1,no_bus))).T,                      (H1.T@(C - V@cp.diag(b)@A)).T,  (V@cp.diag(b0)@A).T@(C+V@cp.diag(b)@A) + (C+V@cp.diag(b)@A).T@(V@cp.diag(b0)@A) - (V@cp.diag(b0)@A).T@(V@cp.diag(b0)@A)]
#             ]  
#                 )>> 0)

#         prob = cp.Problem(objective, constraints)

#         # result = prob.solve(verbose = False, solver = 'MOSEK')

#         try:
#             result = prob.solve(verbose = False, solver = 'MOSEK')
#         except:
#             # Not find the result, return the last run result
#             # print('Break')
#             return b0, obj_pre

#         # print(f'omega: {omega.value}')

#         # Update b0
#         b0 = b.value
        
#         # Terminate condition
#         if np.abs(obj_pre - omega.value) <= tol or ite == max_ite:
#             break
        
#         obj_pre = omega.value

#     return b.value, omega.value




