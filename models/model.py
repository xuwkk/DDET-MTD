"""
Contains the LSTM-AE model for anomaly detection and following recovery.
code modified from
https://github.com/hellojinwoo/TorchCoder/blob/master/autoencoders/rae.py
"""


import torch
import torch.nn as nn
import sys
from models.early_stopping import EarlyStopping
from time import time
from configs.nn_setting import nn_setting
torch.manual_seed(0)

"""
ENCODER
"""

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.sample_length = nn_setting['sample_length']
        self.no_feature = nn_setting['feature_size']
        self.latten_dim = nn_setting['lattent_dim']    
        self.no_layer = nn_setting['no_layer']

        # Container of different LSTM layers
        LSTM_layers = []
        output_size = self.no_feature
        for i in range(self.no_layer):
            input_size = output_size
            hidden_size = int(self.no_feature - (self.no_feature - self.latten_dim)/(self.no_layer)*(i+1))
            LSTM = nn.LSTM(
                input_size = input_size,
                hidden_size = hidden_size,
                num_layers = 1,
                batch_first = True
            )
            LSTM_layers.append(LSTM)
            output_size = hidden_size
        self.LSTM_layers = nn.ModuleList(LSTM_layers)
    
    def forward(self,x):
        for idx, layer in enumerate(self.LSTM_layers):
            x, (hidden_state,cell_state) = layer(x)

        return hidden_state.reshape(-1,1,self.latten_dim) # (1,batch_size,hidden_size) -> (batch_size, 1, embedding_size)

"""
DECODER
"""

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.sample_length = nn_setting['sample_length']
        self.no_feature = nn_setting['feature_size']
        self.latten_dim = nn_setting['lattent_dim']    
        self.no_layer = nn_setting['no_layer']

        # Container of different LSTM layers
        LSTM_layers = []
        output_size = self.latten_dim
        for i in range(self.no_layer):
            input_size = output_size
            hidden_size = int(self.latten_dim + (self.no_feature - self.latten_dim)/(self.no_layer)*(i))
            LSTM = nn.LSTM(
                input_size = input_size,
                hidden_size = hidden_size,
                num_layers = 1,
                batch_first = True
            )
            LSTM_layers.append(LSTM)
            output_size = hidden_size
        self.LSTM_layers = nn.ModuleList(LSTM_layers)
        self.output_layer = nn.Linear(hidden_size, self.no_feature)

    def forward(self,x):
        x = x.repeat(1, self.sample_length,1)  # repeat the hidden representation from encoder by sample length
        for idx, layer in enumerate(self.LSTM_layers):
            x, (hidden_state,cell_state) = layer(x)

        return self.output_layer(x)


class LSTM_AE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        
        # Training related parameteres
        self.epochs = nn_setting['epochs']
        self.lr = nn_setting['lr']
        self.patience = nn_setting['patience']
        self.delta = nn_setting['delta']
        self.path = nn_setting['model_path']
        self.device = nn_setting['device']
    
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded
    
    def fit(self, train_loader, valid_loader):
        """
        train_loader/test_loader: pytorch dataloader
        """
        
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        criterion = nn.MSELoss(reduction='mean')
        
        #lattent_weight = nn_setting['lattent_weight']
        
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, delta=self.delta, path=self.path)
        
        num_batches_train = len(train_loader)  # Number of batches
        num_batches_valid = len(valid_loader)
        
        # Initlization
        self.apply(init_weights)
        
        """
        Iterations
        """
        print('Start Training...')
        print('*'*60)
        
        
        for epoch in range(1, self.epochs+1):
            
            start_time = time()
            '''
            Train
            '''
            self.train()  # Allow dropout
            train_loss = 0
            recons_loss = 0
            lattent_loss = 0
            
            for batch_idx, (input, output) in enumerate(train_loader):
                
                # Pass model and data to cuda
                zeros = torch.zeros((input.shape[0], 1, nn_setting['lattent_dim']), device=nn_setting['device'])  # The target of hidden space
                self.to(self.device)
                input, output = input.to(self.device), output.to(self.device)  

                # Compute prediction error
                encoded, decoded = self(input)
                #print(encoded.shape)  # (batch_size, 1, hidden_size)
                loss_recons = criterion(decoded, output)
                #loss_lattent = criterion(encoded, zeros)
                
                loss = loss_recons
                #loss = loss_recons + loss_lattent * lattent_weight
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                recons_loss += loss_recons.item()
                #lattent_loss += loss_lattent.item()
                
            #print(f'Recons Loss: {round(recons_loss/num_batches_train,5)}')
            #print(f'Lattent Loss: {round(lattent_loss/num_batches_train,5)}')
            """
            Validation
            """
            self.eval() # no dropout
            valid_loss = 0
            
            with torch.no_grad(): # don't calculate the gradient
                for input, output in valid_loader:
                    #assert np.all(input.numpy() == output.numpy())
                    zeros = torch.zeros((input.shape[0], 1, nn_setting['lattent_dim']), device=nn_setting['device'])  # The target of hidden space

                    input, output = input.to(self.device), output.to(self.device)
                    encoded, decoded = self(input)
                    loss_recons = criterion(decoded, output)
                    #loss_lattent = criterion(encoded, zeros)
                    #loss = loss_recons + loss_lattent * lattent_weight
                    loss = loss_recons
                    valid_loss += loss.item()
                    
            print(f'Train Loss: {round(train_loss/num_batches_train,7)}')
            print(f'Valid Loss: {round(valid_loss/num_batches_valid,7)}')
            
            end_time = time()
            
            """
            Early stopping
            """
            # early_stopping(valid_loss/num_batches_valid, self)  
            early_stopping(train_loss/num_batches_train, self)   # Using training loss
            
            # Display time
            print(f'Epoch {epoch} time {end_time-start_time}')
            print('*'*60)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
        # load the last checkpoint with the best model
        self.load_state_dict(torch.load(self.path))

# Parameter initialization
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.2, 0.2)   
"""
Training
"""

if __name__ == '__main__':
    import numpy as np
    from configs.config import sys_config
    from torch.utils.data import DataLoader
    import sys
    sys.path.append("./")
    from models.dataset import rnn_dataset
    
    # Load data
    z_noise_summary = np.load(f'gen_data/{sys_config["case_name"]}/z_noise_summary.npy')
    
    print(f'Data shape {z_noise_summary.shape}')
    
    lstm_ae = LSTM_AE()
    lstm_ae.apply(init_weights)    # Use the random initialized weight
    print(lstm_ae)
    
    pytorch_total_params = sum(p.numel() for p in lstm_ae.parameters())
    pytorch_total_params_train = sum(p.numel() for p in lstm_ae.parameters() if p.requires_grad)
    
    print(f'The total number of parameters is {pytorch_total_params}')
    print(f'The total trainable number of parameters is {pytorch_total_params_train}')
    
    device= "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    # Training set
    train_set = rnn_dataset(mode = 'train', istransform=True)
    train_loader = DataLoader(train_set, 
                            batch_size=nn_setting['batch_size'], 
                            shuffle=True)
    
    # Valid set
    valid_set = rnn_dataset(mode = 'valid', istransform=True)
    valid_loader = DataLoader(valid_set, 
                            batch_size=nn_setting['batch_size'], 
                            shuffle=False)
    
    print(f'Train set size: {len(train_loader.dataset)}')
    print(f'Valid set size: {len(valid_loader.dataset)}')
    
    lstm_ae.fit(train_loader, valid_loader)