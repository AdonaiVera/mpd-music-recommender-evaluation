import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

class DataRBM:
    def __init__(self, playlist_track_file_path):
        self.raw = pd.read_csv(playlist_track_file_path)
        self.scaler = StandardScaler()
        torch.manual_seed(42)

    def preprocess(self, train_test_ratio = 0.7, can_convert_to_sparse_tensor = False):
        self.data = pd.pivot_table(self.raw, index='pid', columns='track_id', aggfunc='size', fill_value=0).reset_index()
        # self.all_tracks = self.data.columns

        _, self.test_data = train_test_split(self.data, train_size=train_test_ratio, random_state=42)

        # self.data_matrix = self.data.drop('pid', axis=1).values
        _ = _.drop('pid', axis=1).values
        self.test_matrix = self.test_data.drop('pid', axis=1).values
        
        if can_convert_to_sparse_tensor:
            self.train_preprocessed = self.convertToSparseTensor(_)  # Convert to PyTorch tensor
            self.test_preprocessed = self.convertToSparseTensor(self.test_matrix)  # Convert to PyTorch tensor
        else:
            self.train_preprocessed = torch.tensor(_, dtype=torch.float32)  # Convert to PyTorch tensor
            self.test_preprocessed = torch.tensor(self.test_matrix, dtype=torch.float32)  # Convert to PyTorch tensor
            
    def convertToSparseTensor(self, dense_matrix):
        indices = torch.nonzero(torch.tensor(dense_matrix), as_tuple=False).T  # Shape (2, N)
        values = torch.tensor(dense_matrix[indices[0], indices[1]])  # Get the corresponding non-zero values
        
        # Convert to a sparse tensor (COO format: indices, values, shape)
        return torch.sparse_coo_tensor(indices, values, dense_matrix.shape)