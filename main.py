from models.graph_training_pipeline import GraphTrainingPipeline
import torch
import time

config = {
    "playlist_embedding_dim": 64,
    "song_embedding_dim": 64,
    "dropout": 0.5,
    "learning_rate": 0.01,
    "num_epochs": 200,
    "pos_edge_weight": 2.0,
    "neg_edge_weight": 1.0,
    "optimizer": "adam",
}

# Automatically determine the device
device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = GraphTrainingPipeline(cur_dir="data/", model_arch="GraphSAGE", device=device, config=config)

# Measure training time
start_time = time.time()

pipeline.train()

end_time = time.time()

# Print elapsed time
elapsed_time = end_time - start_time
print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes).")
