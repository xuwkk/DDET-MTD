"""
A simple realization of MATPOWER AC state estimation using PyPower
which can be used alongside with Python based power system steady state analysis
Sources:
    1. MATPOWER: https://github.com/MATPOWER/matpower
    2. MATPOWER SE: https://github.com/MATPOWER/mx-se
    3. PYPOWER: https://github.com/rwl/PYPOWER

The original repo for this code can be found in my GitHub: https://github.com/xuwkk/steady-state-power-system

Some code is removed if not related to the this task

Author: W XU
"""

import numpy as np
from pypower.api import *
from pypower.idx_bus import *
from pypower.idx_brch import *
from pypower.idx_gen import *
import scipy
from configs.config_mea_idx import define_mea_idx_noise
from configs.config import se_config, opt
import copy
from scipy.stats.distributions import chi2
import torch
import sys
from configs.nn_setting import nn_setting
from copy import deepcopy

def cartesian_complex_mul(A_real, A_imag, B_real, B_imag):
    """
    Using torch matmul for 2-D matrix product
    C = A * B
    return: the real and imaginary parts of C
    """
    C_real = torch.matmul(A_real, B_real) - torch.matmul(A_imag, B_imag)
    C_imag = torch.matmul(A_imag, B_real) + torch.matmul(A_real, B_imag)

    return C_real, C_imag

class SE:
    def __init__(self, case, noise_sigma, idx, fpr):
        
        """
        case: the instances case by calling from pypower api, e.g. case = case14()
        noise_sigma = A 1D array contains the noise std of the measurement, please refer to the format in mea_idx
        tol: the tolerance on the minimum jacobian matrix norm changes before considered as convegent
        max_it: maximum iteration
        verbose: description settings on the output
        measurement: the measurement given by each measurement type
        idx: the measurement index given by each measurement type, please refer to the format in mea_idx
        
        measurement type (the order matters)
        1. z = [pf, pt, pg, vang, qf, qt, qg, vmag]  (in MATPOWER-SE)
        2. z = [pf, pt, pi, vang, qf, qt, qi, vmag]  (in our current settings)
        In the future version, the selection on 1 and 2 should be added
        """
        
        """
        Define the grid parameter
        """
        
        # Case
        self.case = case
        
        # For analysis only
        # case['branch'][:,TAP] = 0
        
        case_int = ext2int(case)                                                         # Covert the start-1 to start-0 in python                     
        self.case_int = case_int
        
        # Numbers
        self.no_bus = len(case['bus'])
        self.no_brh = len(case['branch'])
        self.no_gen = len(case['gen'])
        self.no_mea = 0
        for key in idx.keys():
            self.no_mea = self.no_mea + len(idx[key])
        
        # Determine the bus type
        self.ref_index, pv_index, pq_index = bustypes(case_int['bus'], case_int['gen'])  # reference bus (slack bus), pv bus, and pq (load bus)
        self.non_ref_index = list(pq_index) + list(pv_index)                             # non reference bus
        self.non_ref_index.sort()
        
        """
        Define matrices related to measurement noise
        """
        self.noise_sigma = noise_sigma                     # std
        self.R = np.diag(noise_sigma**2)                   # R
        self.R_inv = np.diag(1/self.noise_sigma**2)        # R^-1 
        self.R_inv_12 = np.diag(1/self.noise_sigma)        # R^-1/2
        
        self.R_inv_12_ = self.R_inv_12[:self.no_brh,:self.no_brh]  # For the active power flow
        
        DoF = self.no_mea - 2*(self.no_bus - 1)            # Degree of Freedom
        self.bdd_threshold = chi2.ppf(1-fpr, df = DoF)     # BDD detection threshold
        
        # For the power flow measurement system
        DoF_pf = self.no_brh - (self.no_bus - 1)
        self.bdd_threshold_pf = chi2.ppf(1-fpr, df = DoF_pf)
        
        # Hidden threshold
        fpr_attacker = 0.4
        self.bdd_threshold_attacker = chi2.ppf(1-fpr_attacker, DoF)
        
        """
        Incidence Matrix
        """
        
        # Branch Incidence Matrix
        f_bus = case_int['branch'][:, 0].astype('int')        # list of "from" buses
        t_bus = case_int['branch'][:, 1].astype('int')        # list of "to" buses
        self.Cf = np.zeros((self.no_brh,self.no_bus))         # "from" bus incidence matrix
        self.Ct = np.zeros((self.no_brh,self.no_bus))         # "to" bus incidence matrix
        for i in range(self.no_brh):
            self.Cf[i,f_bus[i]] = 1
            self.Ct[i,t_bus[i]] = 1
        
        self.A = self.Cf-self.Ct
        self.Ar = np.delete(self.A, self.ref_index, axis = 1) # Remove the reference bus column
        
        # Generator Incidence Matrix
        self.Cg = np.zeros((self.no_bus,self.no_gen))
        for i in range(self.no_gen):
            self.Cg[int(case_int['gen'][i,0]),i] = 1
        
        # Measurement incidence Matrix
        self.idx = idx
        no_idx_all = 4*self.no_brh + 4*self.no_bus
        self.IDX = np.zeros((self.no_mea, no_idx_all))
        _cache1 = 0
        _cache2 = 0
        for key in idx.keys():
            for _idx, _value in enumerate(idx[key]):
                self.IDX[_cache1+_idx, _cache2+_value] = 1                 
            _cache1 = _cache1 + len(idx[key])
            if key == 'pf' or key == 'pt' or key == 'qf' or key == 'qt':
                _cache2 = _cache2 + self.no_brh
            else:
                _cache2 = _cache2 + self.no_bus
        
        # Calculate the admittance matrix
        self._admittance_matrix()
        
        # Record the original parameters
        self.r = copy.deepcopy(case['branch'][:,BR_R])
        self.x = copy.deepcopy(case['branch'][:,BR_X])
        y = 1/(case['branch'][:,BR_R]+1j*case['branch'][:,BR_X])
        self.g = np.real(y)  # Conductance
        self.b = np.imag(y)  # Suseptance
        self.bc = case['branch'][:,BR_B]
        
        # Tap ratio
        self.tap = np.ones((self.no_brh,))
        tap_idx = np.where(self.case['branch'][:,TAP] != 0)[0]
        self.tap[tap_idx] = self.case['branch'][tap_idx,TAP]

        # Tap phase shift
        self.theta_s = self.case['branch'][:,SHIFT]
    
    def update_reactance(self,x_new):
        """
        Update reactance in self.case
        """
        self.case['branch'][:,BR_X] = x_new
        self.case_int = ext2int(self.case)

        # Update the admittance matrix
        self._admittance_matrix()
    
    def _admittance_matrix(self):
        """
        Calculate the Admittance matrix according to the current admittance in self.case
        """
        Ybus, Yf, Yt = makeYbus(self.case_int['baseMVA'], self.case_int['bus'], self.case_int['branch'])
        self.Ybus = scipy.sparse.csr_matrix.todense(Ybus).getA()
        self.Yf = scipy.sparse.csr_matrix.todense(Yf).getA()
        self.Yt = scipy.sparse.csr_matrix.todense(Yt).getA()

        self.Gsh = self.case['bus'][:,GS]/self.case['baseMVA']
        self.Bsh = self.case['bus'][:,BS]/self.case['baseMVA']
    
    def run_opf(self, verbose = False, **kwargs):
        """
        Run the optimal power flow
        """
        
        case_opf = copy.deepcopy(self.case)
        if 'load_active' in kwargs.keys():
            # if a new load condition is given
            case_opf['bus'][:,PD] = kwargs['load_active']
            case_opf['bus'][:,QD] = kwargs['load_reactive']
        else:
            # Use the default load condition in the case file
            pass
        
        if verbose:
            result = runopf(case_opf)
        else:
            result = runopf(case_opf, opt)
        
        return result
        
    def construct_mea(self, result):
        """
        Given the OPF result, construct the measurement vector
        z = [pf, pt, pi, vang, qf, qt, qi, vmag] in the current setting
        """
        pf = result['branch'][:,PF]/self.case['baseMVA']
        pt = result['branch'][:,PT]/self.case['baseMVA']
        pi = (self.Cg@result['gen'][:,PG] - result['bus'][:,PD])/self.case['baseMVA']
        vang = result['bus'][:, VA]*np.pi/180             # In radian
        
        qf = result['branch'][:,QF]/self.case['baseMVA']
        qt = result['branch'][:,QT]/self.case['baseMVA']
        qi = (self.Cg@result['gen'][:,QG] - result['bus'][:,QD])/self.case['baseMVA']
        vmag = result['bus'][:, VM]
        
        z = np.concatenate([pf, pt, pi, vang, qf, qt, qi, vmag], axis = 0)
        
        z = self.IDX@z   # Select the measurement
        #print(z.shape)
        #print(self.R.shape)
        
        z_noise = z + np.random.multivariate_normal(mean = np.zeros((self.no_mea,)), cov = self.R)
        z = np.expand_dims(z, axis = 1)
        z_noise = np.expand_dims(z_noise, axis = 1)
        
        vang_ref = vang[self.ref_index]
        vmag_ref = vmag[self.ref_index]
        
        return z, z_noise, vang_ref, vmag_ref
    
    def construct_v_flat(self, vang_ref, vmag_ref):
        
        """
        Construct a flat start voltage Given the reference bus voltage
        vmag_ref: the reference bus voltage magnitude from result
        vang: the reference bus voltage phase angle from result
        """
        vang_flat = np.zeros((self.no_bus,))
        vmag_flat = np.ones((self.no_bus,))
        
        vang_flat[self.ref_index] = vang_ref    
        vmag_flat[self.ref_index] = vmag_ref
        
        return vang_flat, vmag_flat
        
    def h_x_pypower(self, v):
        """
        Estimate the measurement from the state: z_est = h(v)
        v is complex power
        z = [pf, pt, pi, vang, qf, qt, qi, vmag] in the current setting
        """
        #print(( np.diag(self.Cf@v)).shape)
        #print((np.conj(self.Yf)).shape)
        #print((np.conj(v).shape))

        # print(np.linalg.norm(self.Yf, 2))
        
        sf = np.diag(self.Cf@v)@np.conj(self.Yf)@np.conj(v)    # "from" complex power flow
        st = np.diag(self.Ct@v)@np.conj(self.Yt)@np.conj(v)    # "to" complex power flow
        si = np.diag(v)@np.conj(self.Ybus)@np.conj(v)          # complex power injection
        vang = np.angle(v)
        vmag = np.abs(v)
        
        pf = np.real(sf)
        pt = np.real(st)
        pi = np.real(si)
        qf = np.imag(sf)
        qt = np.imag(st)
        qi = np.imag(si)

        z_est = np.concatenate([pf, pt, pi, vang, qf, qt, qi, vmag], axis = 0)
        z_est = np.expand_dims(z_est, axis = 1)
        z_est = self.IDX@z_est
        
        return z_est
        
    
    def jacobian(self, v_est):
        """
        Given the stationary state, Calculate the Jacobian matrix.
        Return: the Jacobian matrix of full measurement
        """
        # 
        # Compute the Jacobian matrix
        [dsi_dvmag, dsi_dvang] = dSbus_dV(self.Ybus, v_est)   # si w.r.t. v
        [dsf_dvang, dsf_dvmag, dst_dvang, dst_dvmag, _, _] = dSbr_dV(self.case_int['branch'], self.Yf, self.Yt, v_est)  # sf w.r.t. v

        dpf_dvang = np.real(dsf_dvang)
        dqf_dvang = np.imag(dsf_dvang)
        dpf_dvmag = np.real(dsf_dvmag)
        dqf_dvmag = np.imag(dsf_dvmag)
        
        dpt_dvang = np.real(dst_dvang)
        dqt_dvang = np.imag(dst_dvang)   
        dpt_dvmag = np.real(dst_dvmag)
        dqt_dvmag = np.imag(dst_dvmag)  
        
        dpi_dvang = np.real(dsi_dvang)
        dqi_dvang = np.imag(dsi_dvang)
        dpi_dvmag = np.real(dsi_dvmag)
        dqi_dvmag = np.imag(dsi_dvmag)
        
        dvang_dvang = np.eye(self.no_bus)
        dvang_dvmag = np.zeros((self.no_bus, self.no_bus))
        
        dvmag_dvang = np.zeros((self.no_bus, self.no_bus))
        dvmag_dvmag = np.eye(self.no_bus)

        # z = [pf, pt, pi, vang, qf, qt, qi, vmag] in the current setting
        J = np.block([
            [dpf_dvang,    dpf_dvmag],
            [dpt_dvang,    dpt_dvmag],
            [dpi_dvang,    dpi_dvmag],
            [dvang_dvang,  dvang_dvmag],
            [dqf_dvang,    dqf_dvmag],
            [dqt_dvang,    dqt_dvmag],
            [dqi_dvang,    dqi_dvmag],
            [dvmag_dvang,  dvmag_dvmag]
        ])
        # # Remove the reference bus
        # J = np.block([
        #     [dpf_dvang[:,self.non_ref_index],    dpf_dvmag[:,self.non_ref_index]],
        #     [dpt_dvang[:,self.non_ref_index],    dpt_dvmag[:,self.non_ref_index]],
        #     [dpi_dvang[:,self.non_ref_index],    dpi_dvmag[:,self.non_ref_index]],
        #     [dvang_dvang[:,self.non_ref_index],  dvang_dvmag[:,self.non_ref_index]],
        #     [dqf_dvang[:,self.non_ref_index],    dqf_dvmag[:,self.non_ref_index]],
        #     [dqt_dvang[:,self.non_ref_index],    dqt_dvmag[:,self.non_ref_index]],
        #     [dqi_dvang[:,self.non_ref_index],    dqi_dvmag[:,self.non_ref_index]],
        #     [dvmag_dvang[:,self.non_ref_index],  dvmag_dvmag[:,self.non_ref_index]]
        # ])
        
        J = np.array(J)         # Force convert to numpy array

        return J     # This J is not full column rank as the reference bus is not removed.

    def ac_se_pypower(self, z_noise, vang_ref, vmag_ref, is_honest = True, **kwargs):
        """
        AC-SE based on pypower
        v_initial: initial gauss on the 
        is_honest: 
        If True, then honest state estimation is used where the Jacobian matrix is updated at each iteration
        If False, then dishonest state estimation is used where the Jacobian matrix is fixed at the initial point.
        """
        """
        Verbose
        """
        if len(kwargs.keys()) == 0:
            # Default verbose
            tol = 1e-5,    
            max_it = 100     
            verbose = 0
            
        else:
            tol = kwargs['config']['tol']    
            max_it = kwargs['config']['max_it']
            verbose = kwargs['config']['verbose']
        
        """
        Initialization
        """
        is_converged = 0
        ite_no = 0
        vang_est, vmag_est = self.construct_v_flat(vang_ref, vmag_ref)      # Flat start state
        v_est = vmag_est*np.exp(1j*vang_est)             # (no_bus, )
        
        # For the first run and also the dishonest Jacobian, the below value will never change
        J = self.jacobian(v_est)       # Jacobian matrix on the flat state 
        # Remove the reference columns (for both angle and magnitude)
        J = np.delete(J, [self.ref_index, self.ref_index+self.no_bus], 1)
        J = self.IDX@J          # Select the measurement
        # Update rule: x := x_0 + (Jx0^T * R^-1 * Jx0)^-1 * Jx0^T * R^-1 * (z-h(x_0))
        G = J.T@self.R_inv@J
        G_inv = np.linalg.inv(G)

        """
        Gauss-Newton Iteration
        """
        
        while is_converged == False and ite_no < max_it:
            # Update iteration counter
            ite_no += 1

            # Compute estimated measurement
            z_est = self.h_x_pypower(v_est)       # z is 2D array (no_mea, 1) 
            
            if is_honest == False:
                # No need to update the Jacobian
                #print('not update')
                pass
            else:
                #print('update')
                # Update the Jacobian
                J = self.jacobian(v_est)         # It is repeated for the first run on v flat!
                J = np.delete(J, [self.ref_index, self.ref_index+self.no_bus], 1)
                J = self.IDX@J          # Select the measurement
                # Update rule: x := x_0 + (Jx0^T * R^-1 * Jx0)^-1 * Jx0^T * R^-1 * (z-h(x_0))
                G = J.T@self.R_inv@J
                G_inv = np.linalg.inv(G)
            
            # Test observability
            rankG = np.linalg.matrix_rank(G)
            if rankG < G.shape[0]:
                print(f'The current measurement setting is not observable.')
                break
            
            F = J.T@self.R_inv@(z_noise-z_est)
            dx = (G_inv@F).flatten()  # Note that the voltages are 1D array
            normF = np.linalg.norm(F, np.inf) 
            
            if verbose == 0:
                pass
            else:
                print(f'iteration {ite_no} norm of mismatch: {np.round(normF,6)}')
            
            # Terminate condition
            if normF < tol:
                is_converged = True
            
            # Update
            vang_est[self.non_ref_index] = vang_est[self.non_ref_index] + dx[:len(self.non_ref_index)]
            vmag_est[self.non_ref_index] = vmag_est[self.non_ref_index] + dx[len(self.non_ref_index):]
            v_est = vmag_est*np.exp(1j*vang_est)
        
        return v_est

    def bdd_residual(self, z_noise, v_est):
        """
        Find the residual of chi^2 detector given the estimated state
        """
        # Find z_est
        z_est = self.h_x_pypower(v_est)
        
        return ((z_noise-z_est).T@self.R_inv@(z_noise-z_est))[0,0]
        
    """
    Recovery
    """
    def gen_torch_constant(self):
        """
        Convert the numpy array into torch array for recovery
        """
        self.device = nn_setting['device']
        # print(f'device: {self.device}')
        self.Cf_torch = torch.tensor(self.Cf).float().to(self.device)
        self.Ybus_real_torch = torch.real(torch.tensor(self.Ybus)).float().to(self.device)
        self.Ybus_imag_torch = -torch.imag(torch.tensor(self.Ybus)).float().to(self.device)          # negative because of the conjugate
        self.Yf_real_torch = torch.real(torch.tensor(self.Yf)).float().to(self.device)
        self.Yf_imag_torch = -torch.imag(torch.tensor(self.Yf)).float().to(self.device)              # negative because of the conjugate
        self.R_inv_torch = torch.tensor(self.R_inv).float().to(self.device)
        self.R_inv_12_torch = torch.tensor(self.R_inv_12).float().to(self.device)
    
    def cartesian_state_to_measurement(self, v_real, v_imag):
        '''
        Using torch and cartesian format state to calculate the measurement
        This is further used in reconstruction measurement, 
        and v_real and v_imag should be decision variable
        
        v_real: real part of complex state
        v_imag: imaginary part of complex state
        
        ref: matpower manual: https://matpower.org/docs/MATPOWER-manual.pdf
        '''
        v_diag_real = torch.diag((v_real))
        v_diag_imag = torch.diag((v_imag))
        v_real = v_real.unsqueeze(1)                # extend the dimension by one
        v_imag = -v_imag.unsqueeze(1)               # negative because of the conjugate
        
        # power injections
        v_diag_Ybus_real, v_diag_Ybus_imag = cartesian_complex_mul(v_diag_real, v_diag_imag, self.Ybus_real_torch, self.Ybus_imag_torch)
        P_inj, Q_inj = cartesian_complex_mul(v_diag_Ybus_real,v_diag_Ybus_imag,v_real,v_imag)

        # power flows
        Cf_v_real, Cf_v_imag = torch.matmul(self.Cf_torch,v_real), torch.matmul(self.Cf_torch,-v_imag)         # the minus here is because there is no conjugate here
        Cf_v_diag_real, Cf_v_diag_imag = torch.diag(Cf_v_real.squeeze(1)), torch.diag(Cf_v_imag.squeeze(1))
        Yf_V_real, Yf_V_imag = cartesian_complex_mul(self.Yf_real_torch, self.Yf_imag_torch, v_real, v_imag)
        P_from, Q_from = cartesian_complex_mul(Cf_v_diag_real, Cf_v_diag_imag, Yf_V_real, Yf_V_imag)
        #return v_diag_Ybus_real
        return torch.cat([P_from,P_inj,Q_from,Q_inj], axis = 0)
    
    """
    MTD
    """
    
    # max rank MTD (aka, random MTD)
    def random_mtd(self, x_max_ratio, x_min_ratio = 0.05, dfacts_index = 'full'):
        # the reactance should be inside the range
        # we note that the reactance should not be too small 
        # e.g. x_max_ratio = 0.2
        # e.g. x_min_ratio = 0.05
        # dfacts_index is the branches installed with D-FACTS devices : START FROM 0!!!!!
        # assume all the branches have D-FACTS devices
        
        if dfacts_index == 'full':
            dfacts_index = np.arange(self.no_brh)
        
        pertur_ratio = x_min_ratio + (x_max_ratio - x_min_ratio)*np.random.rand(self.no_brh) # [0.05,0.2] positive
        pos_neg = np.random.choice([-1,1], self.no_brh)                                      # randomly [-0.2,-0.05] or [0.05,0.2]
        pertur_ratio_ = pertur_ratio * pos_neg
        
        x_mtd = deepcopy(self.x)

        x_mtd = x_mtd*(1+pertur_ratio_)
        
        # set 0 to the non D-FACTS devices
        non_dfact_branch = list(set(np.arange(self.no_brh)) - set(dfacts_index))
        x_mtd[non_dfact_branch] = self.x[non_dfact_branch]
        
        assert np.all(np.abs(x_mtd[dfacts_index]-self.x[dfacts_index])/self.x[dfacts_index] >= x_min_ratio) ## test 
        assert np.all(np.abs(x_mtd[dfacts_index]-self.x[dfacts_index])/self.x[dfacts_index] <= x_max_ratio)
        
        return x_mtd
    
    # Construct matrices for optimization problems
    
    def H_v_hat(self, v_est):
        """
        The Jacobian matrix from active power flow to voltage phase angle
        v_est is the previous state (before MTD is triggered)
        """
        
        # Method 1, directly calculate from pypower function dSbr_dV
        [dsf_dvang_fun, _, _, _, _, _] = dSbr_dV(self.case_int['branch'], self.Yf, self.Yt, v_est)  # sf w.r.t. v
        dpf_dvang_fun = np.delete(np.real(dsf_dvang_fun), self.ref_index, axis = 1)
        # Method 2, calculate from the explicit equation
        # vmag = np.abs(v_est)
        # vang = np.angle(v_est)
        # V =  np.diag((self.Cf@vmag) * (self.Ct@vmag))
        # G = np.diag(self.g)
        # B = np.diag(self.b)
        # Ars = np.diag(np.sin(self.A@vang))@self.Ar
        # Arc = np.diag(np.cos(self.A@vang))@self.Ar
        # C = V@G@Ars
        # dpf_dvang_equ = C-V@B@Arc
        
        return np.array(dpf_dvang_fun)
    
    def H_v_hat_robust(self, v_est):
        """
        Generate matrices for MTD effectiveness
        """
        
        vmag = np.abs(v_est)
        vang = np.angle(v_est)
        V =  np.diag((self.Cf@vmag) * (self.Ct@vmag))
        G = np.diag(self.g)
        Ars = np.diag(np.sin(self.A@vang))@self.Ar
        C = V@G@Ars
        
        Arc = np.diag(1/self.tap) @ np.diag(np.cos(self.A@vang))@self.Ar
        
        return V, C, Ars, Arc
    
    def H_v_flat(self, vang_ref, vmag_ref):
        """
        Generate matrices for MTD effectiveness
        """
        
        # Construct the flat voltage
        vang_flat, vmag_flat = self.construct_v_flat(vang_ref, vmag_ref)
        V =  np.diag((self.Cf@vmag_flat) * (self.Ct@vmag_flat))
        G = np.diag(self.g)
        Ars = np.diag(np.sin(self.A@vang_flat))@self.Ar   
        Arc = np.diag(1/self.tap) @ np.diag(np.cos(self.A@vang_flat))@self.Ar   # Consider the influence of tap ratio
        C = V@G@Ars
        
        return V, C, Ars, Arc
    
    def mtd_matrix(self,Jr_N):
        
        #Jr_N: normalized reduced jacobian matrix, calculate the projection matrix
        Pr_N = Jr_N@np.linalg.inv(Jr_N.T@Jr_N)@Jr_N.T
        Sr_N = np.eye(self.no_brh) - Pr_N
        
        return Pr_N, Sr_N
    
    def hidden_matrix(self, v_est):
        
        """
        Matrices for MTD hiddenness.
        Calculate the S_v_hat in the objective of hiddenness, also normalized w.r.t. the measurement noise
        """
        
        """
        S_v_hat_N
        """
        # Calculate the Jacobian matrix
        H_v_est = self.jacobian(v_est=v_est)   # Full Jacobian matrix
        H_v_est = self.IDX@H_v_est             # Selected Jacobian matrix
        H_v_est_N = self.R_inv_12@H_v_est      # Normalization

        # Calculate residual matrix
        S_v_hat_N = np.eye(self.no_mea) - H_v_est_N@np.linalg.inv(H_v_est_N.T@H_v_est_N)@H_v_est_N.T

        """
        H_b_N
        """
        tau = copy.deepcopy(self.tap)
        theta_s = copy.deepcopy(self.theta_s)

        # Derivative of admittance matrix to susceptance (diag)
        delta_Yff = np.diag(1j*1/(tau**2))
        delta_Yft = np.diag(-1j/(tau*np.exp(-1j*theta_s)))
        delta_Ytf = np.diag(-1j*1/(tau*np.exp(1j*theta_s)))
        delta_Ytt = np.diag(1j*np.ones(self.no_brh))

        Cf = copy.deepcopy(self.Cf)
        Ct = copy.deepcopy(self.Ct)
        v_star = np.conjugate(v_est)

        # Power Injections
        dSbus_db_1 = np.diag(Cf@v_star)@np.conjugate(delta_Yff) + np.diag(Ct@v_star)@np.conjugate(delta_Yft)
        dSbus_db_2 = np.diag(Cf@v_star)@np.conjugate(delta_Ytf) + np.diag(Ct@v_star)@np.conjugate(delta_Ytt)
        dSbus_db = np.diag(v_est)@(Cf.T@dSbus_db_1+Ct.T@dSbus_db_2)

        dPbus_db = np.real(dSbus_db)
        dQbus_db = np.imag(dSbus_db)

        # Power Flows
        dSf_db = np.diag(Cf@v_est)@(np.diag(Cf@v_star)@np.conjugate(delta_Yff) + np.diag(Ct@v_star)@np.conjugate(delta_Yft))
        dPf_db = np.real(dSf_db)
        dQf_db = np.imag(dSf_db)

        # Concat: 
        # NOTE: we only write the case when the measurement is composed by 'HALF_RTU': [pf,pi,qf,qi]
        H_b = np.concatenate([dPf_db,dPbus_db,dQf_db,dQbus_db], axis = 0)

        H_b_N = self.R_inv_12@H_b   # Normalization

        H_hid = S_v_hat_N@H_b_N

        return H_hid

    
"""
An example
"""
if __name__ == "__main__":
    case = case14()
    
    # Define measurement idx
    mea_idx, no_mea, noise_sigma = define_mea_idx_noise(case, 'FULL')
    # Instance the state estimation class
    se = SE(case, noise_sigma=noise_sigma, idx=mea_idx, fpr = 0.02)
    
    # Run OPF to get the measurement
    opt = ppoption()              # OPF options
    opt['VERBOSE'] = 0
    opt['OUT_ALL'] = 0
    opt['OPF_FLOW_LIM'] = 1       # Constraint on the active power flow
    result = runopf(case, opt)

    print(result['success'])
    
    # Construct the measurement
    z, z_noise, vang_ref, vmag_ref = se.construct_mea(result) # Get the measurement
    
    # Run AC-SE
    se_config['verbose'] = 1
    v_est, _ = se.ac_se_pypower(z_noise, vang_ref, vmag_ref, config = se_config)
    residual = se.bdd_residual(z_noise, v_est)    
    print(f'BDD threshold: {se.bdd_threshold}')
    print(f'residual: {residual}')
    
    
        
        