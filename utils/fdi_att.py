"""
A python library to generate FDI attack
Inherited from the state estimation tool run_AC_SE
Author: W XU 
"""

from utils.class_se import SE
from pypower.api import *
import numpy as np
import warnings
import sys
from configs.nn_setting import nn_setting
import torch
import random

class FDI(SE):
    # Inherit from the Class SE
    def __init__(self, case, noise_sigma, idx, fpr):
        super().__init__(case, noise_sigma, idx, fpr)
    
    def gen_ran_att(self, z_noise, att_ratio_max):
        """
        Generate a random attack without using the knowledge of model
        att_ratio_max: the maximum change ratio of each measurement
        """
        att_ratio = -att_ratio_max + att_ratio_max*2*np.random.rand(z_noise.shape[0])
        att_ratio = np.expand_dims(att_ratio, axis = 1)
        z_att_noise = z_noise * (1+att_ratio)

        return z_att_noise

    def gen_fdi_att_ang(self, z_noise_new, v_est_attacker, c_actual):
        """
        Given a specific angle attack vector, calculate the attack measurement
        z_noise: (feature_size, 1)

        This function is used to generate attack vector after the trigger of MTD. 

        new: refer to after mtd
        z_noise_new: measurement after mtd
        v_est_attacker: the state estimation by the attacker, based on the new measurement and old model (the same one in hiddenness)
                        or the groud truth state after MTD
        c_actual: the ground truth angle attack vector

        """
        # Voltage magnitude and angle
        vang_est = np.angle(v_est_attacker)
        vmag_est = np.abs(v_est_attacker)
        vang_att = vang_est.copy()
        vmag_att = vmag_est.copy()

        vang_att = vang_att + c_actual
        v_att = vmag_att * np.exp(1j*vang_att)             # Mag is not attacked

        z_est_attacker = self.h_x_pypower(v_est_attacker)  # Attacker's estimation on the measurement
        z_att_est = self.h_x_pypower(v_att)                # Attacked measurement estimation

        z_att_noise = z_noise_new + z_att_est - z_est_attacker

        return z_att_noise


    # Generate random FDI Attack
    def gen_fdi_att(self, z_noise, v_est, ang_no, mag_no, ang_str, mag_str):
        
        """
        Generate FDI Attack
        z_noise: (feature_size, 1)
        """
        
        # Voltage magnitude and angle
        vang_est = np.angle(v_est)
        vmag_est = np.abs(v_est)
        
        vang_att = vang_est.copy()
        vmag_att = vmag_est.copy()
        
        # Attack position
        ang_posi = random.sample(self.non_ref_index, ang_no)
        mag_posi = random.sample(self.non_ref_index, mag_no)
        
        # Raise an error if the reference bus is attacked
        if self.ref_index[0] in ang_posi or self.ref_index[0] in mag_posi:
            warnings.warn('The reference bus is attacked. Consider reconfiguring the attack positions.')
        
        # Ang attack
        plus_minus = -1+2*np.random.randint(2, size=ang_no)
        ang_ratio = (ang_str-0.1 + 0.1*np.random.rand(ang_no))*plus_minus
        vang_att[ang_posi] = vang_est[ang_posi] * (1 + ang_ratio)
        
        # Mag attack
        plus_minus = -1+2*np.random.randint(2, size=mag_no)
        mag_ratio = (mag_str-0.1 + 0.1*np.random.rand(mag_no))*plus_minus
        vmag_att[mag_posi] = vmag_est[mag_posi] * (1 + mag_ratio)
        
        v_att = vmag_att * np.exp(1j*vang_att)
        
        z_att_est = self.h_x_pypower(v_att)
        z_est = self.h_x_pypower(v_est)
        z_att_noise = z_noise + z_att_est - z_est
        
        return z_att_noise, v_att
    
    def gen_fdi_att_dd(self, z_noise, v_est_last, ang_no, mag_no, ang_str, mag_str):
        """
        Generate attack for DD detector detection with state already calculated
        
        z_noise: (1,sample_length,feature_size)
        v_est: (no_bus,)
        """
        # Convert data format
        v_est_last = v_est_last.numpy()
        # Extract the last measurement
        z_noise_last = np.expand_dims(z_noise[0,-1].numpy(),1)
        # Generate FDI attack
        z_att_noise_last, v_att_est_last = self.gen_fdi_att(z_noise=z_noise_last, v_est=v_est_last, ang_no=ang_no, mag_no=mag_no, ang_str=ang_str, mag_str=mag_str)
        # print(v_att_est_last)
        # Replace the last measurement into attacked one
        z_att_noise = z_noise.clone()
        z_att_noise[:,-1] = torch.transpose(torch.from_numpy(z_att_noise_last).float(),0,1)
        
        return z_att_noise, v_att_est_last