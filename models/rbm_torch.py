import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.evaluation_metrics import single_eval
from methods.preprocess_rbm import DataRBM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if device.type == "cuda":
    torch.cuda.set_device(int(os.environ["CUDA_VISIBLE_DEVICES"]))
    # Limit memory growth on the GPU
    # torch.cuda.set_per_process_memory_fraction(0.5, device=int(os.environ["CUDA_VISIBLE_DEVICES"]))  # Limit memory usage to 50% of the total GPU memory on device 0


class RBM(nn.Module):
    def __init__(self, visible_units, hidden_units):
        super(RBM, self).__init__()
        
        # Weight matrix (visible to hidden)
        self.W = nn.Parameter(torch.randn(visible_units, hidden_units) * 0.1)
        # Bias for visible units
        self.b = nn.Parameter(torch.zeros(visible_units))
        # Bias for hidden units
        self.c = nn.Parameter(torch.zeros(hidden_units))
        
    def sample_h(self, v):
        """Sample hidden layer given visible layer."""
        h_prob = torch.sigmoid(torch.matmul(v, self.W) + self.c)  # Compute probability
        h_sample = torch.bernoulli(h_prob)  # Sample using Bernoulli distribution
        return h_sample
    
    def sample_v(self, h):
        """Sample visible layer given hidden layer."""
        v_prob = torch.sigmoid(torch.matmul(h, self.W.t()) + self.b)  # Compute probability
        v_sample = torch.bernoulli(v_prob)  # Sample using Bernoulli distribution
        return v_sample
    
    def forward(self, v):
        """Forward pass: Computes hidden layer from visible layer."""
        h = self.sample_h(v)
        return h
    

class RBMHandler:
    def __init__(self, visible_units, device, dataHandler, hidden_units = 100, learning_rate = 0.01, epochs = 10, batch_size = 32):
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.dataHandler = dataHandler

        # Create the RBM model
        self.rbm = RBM(self.visible_units, self.hidden_units).to(device)
        print("RBM INIT")

        self.optimizer = optim.SGD(self.rbm.parameters(), lr=self.learning_rate)
        print("Optimiser INIT")

    def contrastiveDivergence(self, rbm, v, k=1):
        # Step 1: Positive phase (data -> hidden)
        h0 = rbm.sample_h(v)
        
        # Step 2: Negative phase (reconstruct v from h, then h' from v')
        v_ = rbm.sample_v(h0)  # Reconstruct visible layer
        h_ = rbm.sample_h(v_)  # Reconstruct hidden layer
        for _ in range(k-2):    # Iterate k times for contrastive divergence
            v_ = rbm.sample_v(h_)  # Reconstruct visible layer again
            h_ = rbm.sample_h(v_)  # Reconstruct hidden layer
        
        v_ = rbm.sample_v(h_)  # Reconstruct visible layer again
        # Step 3: Calculate the gradients
        positive_grad = torch.matmul(v.t(), h0)
        negative_grad = torch.matmul(v_.t(), h_)
        
        return positive_grad, negative_grad

    def trainModel(self, training_data):
        self.training_data = training_data        
        self.training_dataset = TensorDataset(self.training_data)
        dataloader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            epoch_loss = 0
            for data in dataloader:
                v = data[0].to(device)  # Data is a batch of playlist-track matrix rows
                
                # Compute Contrastive Divergence
                positive_grad, negative_grad = self.contrastiveDivergence(self.rbm, v, k=1)
                
                # Compute the gradient of the loss
                loss = torch.mean((positive_grad - negative_grad) ** 2)  # Reconstruction error
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()

                torch.cuda.empty_cache()

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss / len(dataloader)}")

    def validateModel(self, test_data, test_batch_size = 0):
        batch_size = self.batch_size if test_batch_size == 0 else test_batch_size
        dataset_reconstruction = TensorDataset(test_data)
        dataloader_reconstruction = DataLoader(dataset_reconstruction, batch_size=batch_size, shuffle=False)

        reconstructed_scores_list = []
        # Perform reconstruction with smaller batches
        with torch.no_grad():
            epoch_loss = 0

            for data in dataloader_reconstruction:
                v = data[0].to(self.device)  # Move the batch to GPU
                # Perform forward pass (reconstruction)
                reconstructed_batch = self.rbm.sample_v(self.rbm.sample_h(v))
                # Compute loss or other metrics here if necessary
                # Example loss computation (optional):
                loss = torch.mean((v - reconstructed_batch) ** 2)  # Reconstruction error
                epoch_loss += loss.item()
                reconstructed_scores_list.append(reconstructed_batch.cpu())

            print(f"Reconstruction error: {epoch_loss / len(dataloader_reconstruction)}")

        self.reconstructed_scores = torch.cat(reconstructed_scores_list, dim=0).numpy() # Reconstruct the playlist matrix
        torch.cuda.empty_cache() # Clean up GPU memory after reconstruction


    # Get ground truth (relevant tracks) for each user (playlist)
    def get_ground_truth_for_user(self, playlist_id, df):
        """Returns the ground truth (tracks) for a given playlist (user)."""
        ground_truth = df[df['pid'] == playlist_id]['track_id'].values.tolist()
        return ground_truth

    def evaluateModel(self):
        # Wrap the playlist tensor into a DataLoader
        DF = self.dataHandler.raw
        batch_size = 16
        playlist_ids = DF['pid'].unique()
        playlist_tensor = torch.tensor(playlist_ids, dtype=torch.long)  # Tensor of playlist IDs
        playlist_dataset = TensorDataset(playlist_tensor)  # Create a dataset from the playlist IDs
        playlist_dataloader = DataLoader(playlist_dataset, batch_size=batch_size, shuffle=False)
        RECONSTRUCTED_SCORES = self.reconstructed_scores
        # Loop through each batch in the DataLoader and evaluate
        with torch.no_grad():
            all_rprecision = []

            for data in playlist_dataloader:
                # Get the playlist_ids for the current batch
                batch_playlist_ids = data[0].numpy()  # Playlist IDs for this batch
                
                # Extract the scores for the batch
                scores_batch = RECONSTRUCTED_SCORES[batch_playlist_ids]
                
                # Extract the ground truth for the batch (e.g., the seed tracks)
                seeds_batch = [self.get_ground_truth_for_user(pid, DF) for pid in batch_playlist_ids]

                rprecision_batch = []
                for scores, seeds, playlist_id in zip(scores_batch, seeds_batch, batch_playlist_ids):
                    # Get the ground truth and class labels for the user
                    answer = self.get_ground_truth_for_user(playlist_id, DF)
                    rprecision = single_eval(scores, seed=seeds, answer=answer)
                    rprecision_batch.append(rprecision)

                all_rprecision.extend(rprecision_batch)
            
            print(f"Overall R-Precision: {rprecision}")
        
        torch.cuda.empty_cache() # Clean up GPU memory after evaluation

    def predict(self, data):
        # Generate hidden layer activations for a given playlist
        hidden_activations = self.rbm.sample_h(data.to(device))

        # Recommend tracks (for example, based on which hidden units have the highest activation)
        recommended_tracks = hidden_activations[0].topk(10).indices
        print(f"Recommended tracks: {recommended_tracks}")
        return recommended_tracks




DATA_HANDLER = DataRBM(playlist_track_file_path="data/processed/playlist_tracks_df.csv")
DATA_HANDLER.preprocessForTorch()
MATRIX_DATA = DATA_HANDLER.preprocessed_torch
MATRIX_DATA = MATRIX_DATA.to(device)

visible_units = MATRIX_DATA.shape[1]
# Hyperparameters

rbmHandler = RBMHandler(visible_units=visible_units, device=device, dataHandler=DATA_HANDLER)
rbmHandler.trainModel(MATRIX_DATA)

rbmHandler.validateModel(MATRIX_DATA, 16)

DF = DATA_HANDLER.raw

rbmHandler.evaluateModel()
rbmHandler.predict(data=MATRIX_DATA)
