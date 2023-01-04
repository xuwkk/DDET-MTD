"""
TO DETECT AND RECOVER USING LSTM-AE
"""
import torch
import sys
from models.model import LSTM_AE
from models.dataset import rnn_dataset
from torch.utils.data import DataLoader
from configs.nn_setting import nn_setting
import numpy as np
from torch import optim
from models.dataset import scaler
from time import time


class Evaluation:
    def __init__(self, case_class):

        self.nn_setting = nn_setting
        self.case_class = case_class
        self.case_class.gen_torch_constant()                  # Convert the numpy array into torch array for recovery purpose
        
        # Recovery parameter
        self.recover_lr = nn_setting['recover_lr']
        self.mode = nn_setting['mode']
        self.beta_mag = nn_setting['beta_mag']                # Hard Mag loss
        self.beta_real = nn_setting['beta_real']
        self.beta_imag = nn_setting['beta_imag']
        self.max_step_size = nn_setting['max_step_size']
        self.min_step_size = nn_setting['min_step_size']
        
        # Load the model and trained parameters
        model = LSTM_AE()
        model.load_state_dict(torch.load(nn_setting['model_path'], map_location=torch.device(nn_setting['device'])))
        device = nn_setting['device']
        print(f"Using {device} device")
        model.to(device)
        model.eval()
        print(model)

        self.model = model
        self.device = device

        # Load Validation dataset
        # Valid set
        valid_set = rnn_dataset(mode = 'valid', istransform=True)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)
        
        # Calculate validation loss
        loss_fn = torch.nn.MSELoss()
        valid_loss = []
        with torch.no_grad():
            for input, output in valid_loader:
                assert np.all(input.numpy() == output.numpy())
                input, output = input.to(device), output.to(device)
                encoded, decoded = model(input)
                loss = loss_fn(decoded, output)
                valid_loss.append(loss.item())

        # detection threshold
        quantile = [0.80,0.82,0.84,0.86,0.88,0.90,0.92,0.94,0.96,0.98,0.999]
        ae_threshold = np.quantile(valid_loss, quantile)
        self.ae_threshold = ae_threshold
        
        self.quantile_idx = 6   # The index of threshold quantile in evaluation
        
        print(f'The LSTM-AE detection threshold: {self.ae_threshold}')
        
        # Scaling factor
        scaler_ = scaler()
        self.min = torch.from_numpy(scaler_.min).float().to(nn_setting['device']).requires_grad_(False)
        self.max = torch.from_numpy(scaler_.max).float().to(nn_setting['device']).requires_grad_(False)
    
        self.quantile = quantile
    
    def evaluate(self, test_data):
        if torch.is_tensor(test_data):
            test_data = test_data.float().to(self.device)
        else:
            test_data = torch.tensor(test_data, dtype=torch.float32, device=self.device)    # convert to tensor
            
        assert test_data.shape == (1, self.nn_setting['sample_length'], self.nn_setting['feature_size'])
        zeros = torch.zeros((1, 1, self.nn_setting['lattent_dim']), device=self.nn_setting['device'])
        loss_fn = torch.nn.MSELoss()
        self.model.eval()
        with torch.no_grad():
            encoded, decoded = self.model(test_data)
            loss_recons = loss_fn(decoded, test_data)
            loss_lattent = loss_fn(encoded, zeros)
        
        return encoded.cpu().numpy(), decoded.cpu().numpy(), loss_lattent.item(), loss_recons.item()
    
    def construct_v(self, v):
        """
        Seperate the real and imag part voltage for recovery, the reference bus state is not changed
        """
        vreal = torch.real(v)
        vimag = torch.imag(v)
        vreal_1 = vreal[:self.case_class.ref_index[0]].float().to(self.device).requires_grad_(True)  # dim = 1
        vreal_2 = vreal[self.case_class.ref_index[0]+1:].float().to(self.device).requires_grad_(True)
        vimag_1 = vimag[:self.case_class.ref_index[0]].float().to(self.device).requires_grad_(True)  # dim = 1
        vimag_2 = vimag[self.case_class.ref_index[0]+1:].float().to(self.device).requires_grad_(True)

        return vreal_1, vreal_2, vimag_1, vimag_2
    
    def recover_no_physics(self, attack_batch):
        """
        attack_batch: (1, sample_length, feature_size) NOT scaled
        """
        
        # attack identification (normality recovery) without physics constraint
        # Generate constants
        #self.case_class.gen_torch_constant()                 # prepare the tensor constant
        attack_batch = attack_batch.float().to(self.device)
        loss_fn = torch.nn.MSELoss()
        self.model.train()                                    # cudnn RNN backward can only be called in training mode
        
        # construct the measurement vector
        if self.mode == 'pre':
            # use the previous measurement as warm start
            z_est = attack_batch[:, -2:-1, :].clone().float().to(self.device).requires_grad_(True)
        elif self.mode == 'last':
            # use the last measurement as warm start
            z_est = attack_batch[:, -1:, :].clone().float().to(self.device).requires_grad_(True)
        else:
            print(f'Mode error...')
        
        # Using Adam optimizer
        optimizer = optim.Adam([z_est], lr = self.recover_lr)       # Watch on the non reference bus

        # Recovery losses
        loss_recover_summary = []             # Reconstruction loss
        
        start_time = time()
        for step in range(self.max_step_size):
            
            optimizer.zero_grad()
            # Concate
            # NOTE: attack_batch is not scaled
            recover_batch = torch.cat([attack_batch[:,:-1], z_est], dim = 1)
            #recover_batch = recover_batch      # add the one dimension (sample_length,feature_size) ==> (1,sample_length,feature_size)

            # Scale
            recover_batch_scaled = (recover_batch - self.min)/(self.max - self.min + 1e-7)
            
            # Pass the recovered batch into model
            encoded, decoded = self.model(recover_batch_scaled)
            
            loss_recover = loss_fn(decoded, recover_batch_scaled)    # l2 norm
            # does not add penalization loss
            
            loss_recover.backward()
            # print(loss_recover.item())
            optimizer.step()
            
            # Log
            if step >= 1:
                # The zero run give sparse loss=0
                loss_recover_summary.append(loss_recover.item())
            
            # Stop condition
            # bypass the DDD and the iteration step is larger than the minimum state
            # Reduce the threshold a little bit
            if loss_recover.item() <= self.ae_threshold[self.quantile_idx]*0.8 and step >= self.min_step_size: 
                break
        
        end_time = time()
        # Output on cpu for analysis        
        z_recover = z_est.detach().cpu()
        
        recover_time = end_time - start_time
        
        return z_recover, loss_recover_summary, recover_time
        
    def recover(self, attack_batch, v_pre, v_last):
        """
        attack_batch: (1, sample_length, feature_size) NOT scaled
        v_pre: the previous state
        v_last: the last (possibly the attacked) state
        mode: either 'pre' or 'current' as the warm start. 'pre': start from the previous state; 'current': start from the current (possibly attacked) state
        """
        
        # Generate constants
        #self.case_class.gen_torch_constant()                 # prepare the tensor constant
        attack_batch = attack_batch.float().to(self.device)
        loss_fn = torch.nn.MSELoss()
        self.model.train()                                    # cudnn RNN backward can only be called in training mode

        # Warm start
        # v = [before ref, ref, after ref]
        
        # Reference bus based on the last measurement: do not change!
        vreal_ref = torch.real(v_last)[self.case_class.ref_index].float().to(self.device).requires_grad_(False)
        vimag_ref = torch.imag(v_last)[self.case_class.ref_index].float().to(self.device).requires_grad_(False)
        
        # Use the previous state as warm start
        vreal_1_pre, vreal_2_pre, vimag_1_pre, vimag_2_pre = self.construct_v(v_pre)

        # Use the last state as warm start
        vreal_1_last, vreal_2_last, vimag_1_last, vimag_2_last = self.construct_v(v_last)
        
        # Reference in sparse loss based on the last measurement
        vreal_last = torch.cat([vreal_1_last, vreal_ref, vreal_2_last], dim = 0)
        vimag_last = torch.cat([vimag_1_last, vimag_ref, vimag_2_last], dim = 0)
        vreal_input = torch.clone(vreal_last).detach()   
        vimag_input = torch.clone(vimag_last).detach()
        
        # Reference in voltage magnitude based on the last measurement
        v_abs = torch.abs(v_last).to(self.device).float().detach()  

        if self.mode == 'pre':
            vreal_1 = vreal_1_pre
            vreal_2 = vreal_2_pre
            vimag_1 = vimag_1_pre
            vimag_2 = vimag_2_pre
        elif self.mode == 'last':
            vreal_1 = vreal_1_last
            vreal_2 = vreal_2_last
            vimag_1 = vimag_1_last
            vimag_2 = vimag_2_last
        else:
            print(f'Mode error...')
        
        # Using Adam optimizer
        optimizer = optim.Adam([vreal_1, vreal_2, vimag_1, vimag_2], lr = self.recover_lr)       # Watch on the non reference bus
        
        # Recovery losses
        loss_recover_summary = []             # Reconstruction loss
        loss_sparse_real_summary = []         # The real state should not be changed a lot
        loss_sparse_imag_summary = []         # The imag state should not be changed a lot
        loss_v_mag_summary = []               # The loss on the voltage magnitude change
        loss_summary = []                     # Weighted loss 

        start_time = time()
        for step in range(self.max_step_size):
            
            optimizer.zero_grad()

            # Estimated last measurement
            v_real = torch.cat([vreal_1, vreal_ref, vreal_2], dim = 0)
            v_imag = torch.cat([vimag_1, vimag_ref, vimag_2], dim = 0)
            z_est = self.case_class.cartesian_state_to_measurement(v_real, v_imag)   # (feature_size,1)  

            # Concate
            # NOTE: attack_batch is not scaled
            recover_batch = torch.cat([attack_batch[0,:-1], z_est.t()], dim = 0)
            recover_batch = recover_batch.unsqueeze(0)                               # add the one dimension (sample_length,feature_size) ==> (1,sample_length,feature_size)

            # Scale
            recover_batch_scaled = (recover_batch - self.min)/(self.max - self.min + 1e-7)
            
            # Pass the recovered batch into model
            encoded, decoded = self.model(recover_batch_scaled)
            
            # Losses:
            loss_recover = loss_fn(decoded, recover_batch_scaled)    # l2 norm
            loss_sparse_real = torch.norm(v_real - vreal_input, 1)   # l1 norm    
            loss_sparse_imag = torch.norm(v_imag - vimag_input, 1)   # l1 norm
            loss_v_mag = torch.norm(torch.sqrt(torch.square(v_real) + torch.square(v_imag)) - v_abs, p=1)
            
            # Construct loss
            loss = loss_recover + self.beta_real*loss_sparse_real + self.beta_imag*loss_sparse_imag + self.beta_mag*loss_v_mag
            
            # backward 
            loss.backward()
            optimizer.step()
            
            # Log
            if step >= 1:
                # The zero run give sparse loss=0
                loss_recover_summary.append(loss_recover.item())
                loss_sparse_real_summary.append(loss_sparse_real.item())
                loss_sparse_imag_summary.append(loss_sparse_imag.item())
                loss_v_mag_summary.append(loss_v_mag.item())
                loss_summary.append(loss.item())
            
            # Stop condition
            # bypass the DDD and the iteration step is larger than the minimum state
            # Reduce the threshold a little bit
            if loss_recover.item() <= self.ae_threshold[self.quantile_idx]*0.8 and step >= self.min_step_size: 
                break
        
        end_time = time()
        # Output on cpu for analysis
        v_real = v_real.detach().cpu()
        v_imag = v_imag.detach().cpu()
        v_recover = v_real + 1j*v_imag
        
        z_recover = z_est.detach().cpu()
        
        recover_time = end_time - start_time

        return z_recover, v_recover, loss_recover_summary, loss_sparse_real_summary, loss_sparse_imag_summary, loss_v_mag_summary, loss_summary, recover_time