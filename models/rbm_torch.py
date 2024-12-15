import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.evaluation_metrics import single_eval, get_metrics
from methods.preprocess_rbm import DataRBM

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if device.type == "cuda":
    torch.cuda.set_device(int(os.environ["CUDA_VISIBLE_DEVICES"]))
    # Limit memory growth on the GPU
    torch.cuda.set_per_process_memory_fraction(0.5, device=int(os.environ["CUDA_VISIBLE_DEVICES"]))  # Limit memory usage to 50% of the total GPU memory on device 0


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
    def __init__(self, visible_units, device, dataHandler, hidden_units = 100, learning_rate = 0.1, epochs = 200, batch_size = 32):
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

        self.optimizer = optim.adam(self.rbm.parameters(), lr=self.learning_rate)
        print("Optimiser INIT")

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)

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
        if training_data.is_sparse:
            self.training_dataset = TensorDataset(self.training_data.to_dense())
        else:
            self.training_dataset = TensorDataset(self.training_data)
                   
        dataloader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            epoch_loss = 0
            for data in dataloader:
                v = data[0].to(device)  # Data is a batch of playlist-track matrix rows
                
                if v.is_sparse:
                    v = v.to_dense()

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
            self.scheduler.step()

    def validateModel(self, test_data, test_batch_size = 16):
        batch_size = self.batch_size if test_batch_size == 0 else test_batch_size

        if test_data.is_sparse:
            dataset_reconstruction = TensorDataset(test_data.to_dense())
        else:
            dataset_reconstruction = TensorDataset(test_data)
        dataloader_reconstruction = DataLoader(dataset_reconstruction, batch_size=batch_size, shuffle=False)

        reconstructed_scores_list = []
        # Perform reconstruction with smaller batches
        with torch.no_grad():
            epoch_loss = 0

            for data in dataloader_reconstruction:
                v = data[0].to(self.device)  # Move the batch to GPU

                if v.is_sparse:
                    v = v.to_dense()

                # Perform forward pass (reconstruction)
                reconstructed_batch = self.rbm.sample_v(self.rbm.sample_h(v))
                # Compute loss or other metrics here if necessary
                loss = torch.mean((v - reconstructed_batch) ** 2)  # Reconstruction error
                epoch_loss += loss.item()
                reconstructed_scores_list.append(reconstructed_batch)

            print(f"Reconstruction error: {epoch_loss / len(dataloader_reconstruction)}")

        self.reconstructed_scores = torch.cat(reconstructed_scores_list, dim=0).cpu().numpy() # Reconstruct the playlist matrix
        torch.cuda.empty_cache() # Clean up GPU memory after reconstruction


    # Get ground truth (relevant tracks) for each user (playlist)
    def get_ground_truth_for_user(self, playlist_id, df):
        """Returns the ground truth (tracks) for a given playlist (user)."""
        ground_truth = df[df['pid'] == playlist_id]['track_id'].values.tolist()
        return ground_truth

    def evaluateModel(self, GROUND_TRUTH_DATA, DATA_FOR_PLAYLISTS):
        # Wrap the playlist tensor into a DataLoader
        batch_size = 32
        
        # Create a DataLoader for the test dataset
        playlist_tensor = torch.arange(len(DATA_FOR_PLAYLISTS), dtype=torch.long)  # Playlist IDs from 0 to the number of playlists
        playlist_dataset = TensorDataset(playlist_tensor)  # TensorDataset holds the playlist IDs
        playlist_dataloader = DataLoader(playlist_dataset, batch_size=batch_size, shuffle=False)
    
        RECONSTRUCTED_SCORES = torch.tensor(self.reconstructed_scores, dtype=torch.float32)
        # Loop through each batch in the DataLoader and evaluate
        with torch.no_grad():
            all_rprecision = []
            # all_ndcg = []
            # all_rsc = []

            for data in playlist_dataloader:
                # Get the playlist_ids for the current batch
                batch_indices = data[0].numpy()
                
                # Extract the scores for the batch
                scores_batch = RECONSTRUCTED_SCORES[batch_indices]

                rprecision_batch = []
                # ndcg_batch = []
                # rsc_batch = []
                for scores, playlist_index in zip(scores_batch, batch_indices):
                    # Get the ground truth and class labels for the user
                    playlist_id = DATA_FOR_PLAYLISTS.iloc[playlist_index]["pid"]
                    answer = self.get_ground_truth_for_user(playlist_id.item(), GROUND_TRUTH_DATA)
                    rprecision = single_eval(scores, answer=answer)
                    rprecision_batch.append(rprecision)
                    # ndcg_batch.append(ndcg)
                    # rsc_batch.append(rsc)

                all_rprecision.extend(rprecision_batch)
                # all_ndcg.extend(ndcg_batch)
                # all_rsc.extend(rsc_batch)
            
            overall_rprecision = torch.tensor(all_rprecision).mean().item()
            # overall_ndcg = torch.tensor(all_ndcg).mean().item()
            # overall_rsc = torch.tensor(all_rsc).mean().item()
            print(f"Overall R-Precision: {overall_rprecision}") #, NDCG: {overall_ndcg}, RSC: {overall_rsc}")
        
        torch.cuda.empty_cache() # Clean up GPU memory after evaluation

    def predict(self, data):
        # Generate hidden layer activations for a given playlist
        hidden_activations = self.rbm.sample_h(data.to(device))

        # Recommend tracks (for example, based on which hidden units have the highest activation)
        recommended_tracks = hidden_activations[0].topk(10).indices
        print(f"Recommended tracks: {recommended_tracks}")
        return recommended_tracks




DATA_HANDLER = DataRBM(playlist_track_file_path="data/processed/playlist_tracks_df.csv")
DATA_HANDLER.preprocess()
MATRIX_DATA = DATA_HANDLER.train_preprocessed

MATRIX_DATA = MATRIX_DATA.to(device)
visible_units = MATRIX_DATA.shape[1]
# Hyperparameters

rbmHandler = RBMHandler(visible_units=visible_units, device=device, dataHandler=DATA_HANDLER)
rbmHandler.trainModel(MATRIX_DATA)

TEST_MATRIX_DATA = DATA_HANDLER.test_preprocessed
TEST_MATRIX_DATA = TEST_MATRIX_DATA.to(device)
rbmHandler.validateModel(TEST_MATRIX_DATA, 16)

rbmHandler.evaluateModel(GROUND_TRUTH_DATA=DATA_HANDLER.raw, DATA_FOR_PLAYLISTS=DATA_HANDLER.test_data)
rbmHandler.predict(data=TEST_MATRIX_DATA)
