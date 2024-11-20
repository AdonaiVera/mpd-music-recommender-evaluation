from models.graph_training_pipeline import GraphTrainingPipeline
import torch 

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
pipeline.train()