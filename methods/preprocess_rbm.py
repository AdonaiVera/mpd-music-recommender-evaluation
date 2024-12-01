import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
import torch

class DataRBM:
    def __init__(self, playlist_track_file_path):
        self.raw = pd.read_csv(playlist_track_file_path)
        self.scaler = StandardScaler()

    def preprocessForSK(self, normalise = True, sparsify = True):
        self.matrix = pd.pivot_table(self.raw, index='pid', columns='track_id', aggfunc='size', fill_value=0).values
        self.preprocessed_sk = self.matrix
        if normalise:
            self.norm_matrix = self.scaler.fit_transform(self.preprocessed_sk)
            self.sk_preprocessed_sk = self.norm_matrix
        if sparsify:
            self.sparse_matrix_data = csr_matrix(self.preprocessed_sk)
            self.preprocessed_sk = self.sparse_matrix_data

    def preprocessForTorch(self):
        self.matrix = pd.pivot_table(self.raw, index='pid', columns='track_id', aggfunc='size', fill_value=0).values
        self.preprocessed_torch = torch.tensor(self.matrix, dtype=torch.float32)  # Convert to PyTorch tensor
            

