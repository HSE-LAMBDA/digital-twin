import os
import numpy as np
from digital_twin.models.norm_flow.utils.flow import NFModel, DEVICE
from typing import Optional
from tqdm.auto import tqdm
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
import pickle
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from digital_twin.models.norm_flow.utils.transform import LogScaler
from digital_twin.models.norm_flow.utils import metrics
import json
from glob import glob
import pandas as pd
import warnings

pd.options.mode.chained_assignment = None  # default='warn'

current_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINTS_DIR = os.path.join(current_dir, '../checkpoints/')

class NormFlowModel(object):
    def __init__(self, ckpt_dir, n_layers=16, hidden=10):
        """
        Class for cache sampling model. Separate instances are trained for write and read modes
        @param ckpt_dir - model path (for checkpoints save and load)
        @param n_layers - number of model layers (int)
        @param hidden - size of hidden layer (int)
        """
        self.n_layers = n_layers
        self.hidden = hidden
        self.ckpt_dir = ckpt_dir
        self.nf = None # normalizing flow model for generation
        self.conds = ['iodepth', 'n_jobs', 'block_size', 'read_fraction',
                      'depth_by_jobs', 'load_type', 'io_type', 'n_disks', 'r0', 'r1'] # feature (condition) columns used to generate target
        
        self.target = ["iops","lat"] # target columns
    
    def sample(self, X, n_samples=1):
        """
        #n_samples: int, iodepth:int, block_size:int, n_jobs:int, read_fraction:float, device_type:str, io_type:str, load_type:str
        @param n_samples - number of samples to simulate
        @param iodepth
        @param block_size
        @param n_jobs
        @param device_type
        @param io_type
        @param load_type
        Simulate performance for given conditions. Conditions and targets are the same as used in the training stage
        
        Example: 
            model.sample(sample_size=100, iodepth=10, block_size=256, n_jobs=1, read_fraction=55, device_type='nvme', io_type='read', load_type='random')
        """
        
        if isinstance(X, pd.DataFrame):
            iodepth = X['iodepth'].values[0]
            block_size = X['block_size'].values[0]
            read_fraction = X['read_fraction'].values[0]
            io_type = X['io_type'].values[0]
            load_type = X['load_type'].values[0]
            n_jobs = X['n_jobs'].values[0]
            r0 = X['r0'].values[0]
            r1 = X['r1'].values[0]
            n_disks = X['n_disks'].values[0]
        else:
            iodepth = X['iodepth']
            block_size = X['block_size']
            read_fraction = X['read_fraction']
            io_type = X['io_type']
            load_type = X['load_type']
            n_jobs = X['n_jobs']
            r0 = X['r0']
            r1 = X['r1']
            n_disks = X['n_disks']

        depth_by_jobs = iodepth * n_jobs
        conds = {'depth_by_jobs': depth_by_jobs, 'iodepth': iodepth, 'n_jobs': n_jobs, 'n_disks': n_disks,
                 'block_size': block_size, 'read_fraction': read_fraction, 'io_type': io_type, 
                 'load_type': load_type, 'r0': r0, 'r1': r1}
        conds = [[conds[col] for col in self.conds]]*n_samples
        conds = np.array(conds)        
        conds = self.X_scaler.transform(conds)
        y_pred = self.y_scaler.inverse_transform(self.nf.predict(conds))
        return y_pred

    
    def fit(self, X, y, 
            X_val=None, y_val=None, 
            X_log=False, y_log=True, 
            batch_size=200, n_epochs=200, lr=1e-2):
        """
        Method for model training
        @param X - pd.DataFrame of conditions (like "n_jobs", "iodepth")
        @param y - pd.DataFrame of target performance variables (like latency and iops)
        @param X_log - Use log transform for conditions if True
        @param y_log - Use log transform for targets if True
        @param batch_size - batch size for training (int)
        @param n_epochs - number of epoches (int)
        @param lr - learning rate (float)
        """
        print('start training')
        X['depth_by_jobs'] = X['iodepth']*X['n_jobs'] 
        if not all([col in X.columns for col in self.conds]):
            raise Exception(f"Columns {','.join(self.conds)} must present in X")
        if not all([col in y.columns for col in self.target]):
            raise Exception(f"Columns {','.join(self.target)} must present in y")

        X = X[self.conds].values
        y = y[self.target]        
        input_size = X.shape[1]

        if self.nf is None:
            self.nf = NFModel({'var_size': y.shape[1], 'cond_size': X.shape[1], 
                               'n_layers': self.n_layers, 'hidden': self.hidden})

        if X_log: self.X_scaler = make_pipeline(LogScaler(), StandardScaler())
        else: self.X_scaler = StandardScaler()
        
        if y_log: self.y_scaler = make_pipeline(LogScaler(), StandardScaler())
        else: self.y_scaler = StandardScaler()
        
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.values)
        
        
        with open(self.ckpt_dir+'/X_scaler.pkl', 'wb') as f:
            pickle.dump(self.X_scaler, f)        
        with open(self.ckpt_dir+'/y_scaler.pkl', 'wb') as f:
            pickle.dump(self.y_scaler, f)
                
        opt = torch.optim.Adam(self.nf.model.parameters(), lr=lr)
        self.nf.train()
        loss_history = []
        metrics_history = []
        tbar = trange(n_epochs, desc='?', leave=True)
        train_loader = DataLoader(TensorDataset(torch.tensor(y_scaled, dtype=torch.float32, device=DEVICE), 
                                       torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)),
                          batch_size=batch_size, shuffle=True)
        metrics_best = {'IOPS_MEAPE': np.infty,
                'Lat_MEAPE': np.infty}

        validate = X_val is not None and y_val is not None
        for epoch in tbar:
            state_dict = {'nf': self.nf.model.state_dict(), 'n_layers': self.n_layers, 
                          'hidden': self.hidden, 'conds': self.conds, 'input_size': input_size,
                           'target': self.target}
            if validate:
                metrics_ = self._eval(X_val, y_val, agg='median')
                metrics_history.append(metrics_)
                for k, v in metrics_.items():
                    if v < metrics_best[k]:
                        print(f'Epoch: {epoch} {k} is improved: {metrics_best[k]:.2f} -> {v:.2f}')
                        metrics_best[k] = v
                        torch.save(state_dict, f'{self.ckpt_dir}/best_{k}.ckpt')
            else:
                torch.save(state_dict, f'{self.ckpt_dir}/last.ckpt')
            for i, (y_batch, x_batch) in enumerate(train_loader):
                y_batch, x_batch = [x.to(DEVICE) for x in [y_batch, x_batch]]
                # caiculate loss
                loss = -self.nf.model.log_prob(y_batch, x_batch)

                # optimization step
                opt.zero_grad()
                loss.backward()
                opt.step()

                # caiculate and store loss
                loss_history.append(loss.detach().cpu())
                
                msg = f"NLL: {loss.detach().cpu():.3f}"
                if validate:
                    msg += f" IOPS_MEAPE: {metrics_best['IOPS_MEAPE']:.2f} Lat_MEAPE: "
                    msg += f"{metrics_best['Lat_MEAPE']:.2f}"
                
                tbar.set_description(msg)
                tbar.refresh()  # to show immediately the update

    def _eval(self, X_test, y_test, bootstrap_size=1, agg=None):
        res = {'IOPS_MEAPE': [], 'Lat_MEAPE': []} # , 'DS_QDA': []

        for conds in X_test[[col for col in self.conds if col not in {'depth_by_jobs'}]].drop_duplicates().to_dict(orient='records'):
            mask = np.ones(X_test.shape[0])
            for k, v in conds.items():
                mask = np.logical_and(mask, X_test[k] == v)
            y_true = y_test[mask][self.target].values
            y_pred = self.sample(conds, n_samples=y_true.shape[0])

            (mu_iops, mu_latency), (std_iops, std_latency) = metrics.mean_estimation_absolute_percentage_error(y_true,
                                                                                                               y_pred,
                                                                                                               n_iters=bootstrap_size)

            #mu_qda, std_qda = metrics.discrepancy_score(y_test, y_pred, model='QDA',
            #                                            n_iters=bootstrap_size)
            res['IOPS_MEAPE'].append((mu_iops, std_iops))
            res['Lat_MEAPE'].append((mu_latency, std_latency))
            #res['DS_QDA'].append((mu_qda, std_qda))
        if agg == 'mean':
            res = {k: np.mean(list(map(lambda x: x[0], v))) for k, v in res.items()}
        elif agg == 'median':
            res = {k: np.median(list(map(lambda x: x[0], v))) for k, v in res.items()}
        return res
    
    @staticmethod
    def load(ckpt_dir):
        """
        Method to load last checkpoints for model
        """
        # Load the last best ckpt
        list_of_files = glob(f'{ckpt_dir}/*.ckpt')
        latest_file = max(list_of_files, key=os.path.getctime)
        state_dict = torch.load(latest_file, map_location=DEVICE)
        
        model = NormFlowModel(ckpt_dir, state_dict['n_layers'], state_dict['hidden'])
        model.conds = state_dict['conds']        
        model.target = state_dict['target']
        input_size = state_dict['input_size']
        model.nf = NFModel({'var_size': len(model.target), 'cond_size': input_size, 
                               'n_layers': state_dict['n_layers'], 'hidden': state_dict['hidden']})
        model.nf.model.load_state_dict(state_dict['nf'])
        
        # Load scalers
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            with open(f'{ckpt_dir}/X_scaler.pkl', 'rb') as f:
                model.X_scaler = pickle.load(f)
            with open(f'{ckpt_dir}/y_scaler.pkl', 'rb') as f:
                model.y_scaler = pickle.load(f)
            
        return model
    
