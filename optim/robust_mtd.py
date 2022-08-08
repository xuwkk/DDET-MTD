"""
The incomplete MTD algorithm
code modified from https://github.com/xuwkk/Robust_MTD/blob/main/utils/mtd_incomplete.py
"""

from scipy.stats.distributions import chi2, ncx2
import numpy as np
from pypower.api import dcopf, ext2int, bustypes, printpf
from scipy.io import loadmat
from numpy.linalg import norm, inv, matrix_rank, svd
from copy import deepcopy
from scipy.optimize import minimize, NonlinearConstraint, Bounds, linprog
import time

def x_to_b(case_class,x):
    return -x/(case_class.r**2+x**2)

"""
Minimize the frobenius norm
"""
class incomplete_fro():
    def __init__(self, case_class, v_est, x_max, x_min, col_constraint):
        
        case_class = deepcopy(case_class)
        self.r = case_class.r
        
        # Pre-MTD matrices
        H1 = case_class.H_v_hat(v_est)
        
        # Post-MTD matrices
        V, C, Ars, Arc = case_class.H_v_hat_robust(v_est)
    
        # Normalization
        self.H1 = case_class.R_inv_12_@H1
        self.C = case_class.R_inv_12_@C
        self.V = case_class.R_inv_12_@V
        self.A = Arc
        
        self.P1, self.S1 = case_class.mtd_matrix(self.H1)   # Projection matrices
        
        # Constraint parameter
        self.x_max = x_max                                                         
        self.x_min = x_min                                                         
        self.nonlinear_constraint_max = col_constraint   # Column constraint is a vector
        
    # loss function
    def fun_loss(self,x):
        
        b = -x/(self.r**2+x**2)   # To suceptance

        # MTD martrices
        H = self.C + self.V@np.diag(b)@self.A   
        P = H@np.linalg.inv(H.T@H)@H.T 
        fro_norm = norm(self.P1 @ P, ord = "fro")    ## calculate the Frobenious norm
        
        return fro_norm
    
    # nonlinear constraint
    def fun_constraint(self,x):
        
        b = -x/(self.r**2+x**2)

        H = self.C + self.V@np.diag(b)@self.A   
        P = H@np.linalg.inv(H.T@H)@H.T 
        
        con = []  
        # find the cos of each bus
        for i in range(H.shape[-1]):
            # the consine of each column of H_N
            con.append(norm(P @ self.H1[:,i:i+1], ord = 2)/norm(self.H1[:,i:i+1], ord = 2))
        
        return con 
        
    def nonlinear_constraint(self):
        
        # construct the NonlinearConstraint class
        return NonlinearConstraint(fun = self.fun_constraint, 
                                    lb = 0, 
                                    ub = self.nonlinear_constraint_max)
    
    # boundary constraint
    def fun_bound(self):
        
        # don't forget to set zero on the non D-FACTS lines
        return Bounds(lb = self.x_min, ub = self.x_max)

class incomplete_l2():
    def __init__(self, case_class, v_est, x_max, x_min, col_constraint, U_k):
        
        case_class = deepcopy(case_class)
        self.r = case_class.r
        
        # Pre-MTD matrices
        H1 = case_class.H_v_hat(v_est)
        
        # Post-MTD matrices
        V, C, Ars, Arc = case_class.H_v_hat_robust(v_est)
    
        # Normalization
        self.H1 = case_class.R_inv_12_@H1
        self.C = case_class.R_inv_12_@C
        self.V = case_class.R_inv_12_@V
        self.A = Arc
        self.P1, self.S1 = case_class.mtd_matrix(self.H1)   # Projection matrices

        
        # constraint parameter
        self.x_max = x_max                                                         
        self.x_min = x_min                                                         
        self.nonlinear_constraint_max = col_constraint
        
        # projector to the intersection subspace
        self.P_k = U_k@U_k.T                                                  
        
    def fun_loss(self,x):
        
        b = -x/(self.r**2+x**2)

        # MTD martrices
        H = self.C + self.V@np.diag(b)@self.A   
        P = H@np.linalg.inv(H.T@H)@H.T 
        l2_norm = norm(self.P1 @ P  - self.P_k, ord = 2)               
        
        return l2_norm
    
    # nonlinear constraint
    def fun_constraint(self,x):
        
        b = -x/(self.r**2+x**2)

        # MTD martrices
        H = self.C + self.V@np.diag(b)@self.A   
        P = H@np.linalg.inv(H.T@H)@H.T 
        
        con = []  
        # find the cos of each bus
        for i in range(H.shape[-1]):
            # the consine of each column of H_N
            con.append(norm(P @ self.H1[:,i:i+1], ord = 2)/norm(self.H1[:,i:i+1], ord = 2))
        
        return con 
        
    def nonlinear_constraint(self):
        
        # construct the NonlinearConstraint class
        return NonlinearConstraint(fun = self.fun_constraint, 
                                    lb = 0, 
                                    ub = self.nonlinear_constraint_max)
    
    def fun_bound(self):
        # don't forget to set zero on the non D-FACTS lines
        return Bounds(lb = self.x_min, ub = self.x_max)