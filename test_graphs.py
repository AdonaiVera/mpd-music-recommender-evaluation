import torch
import numpy as np
import matplotlib.pyplot as plt
from models.graph_training_pipeline import GraphSAGELinkPrediction
from methods.data_utils import DataManager
from evaluation.visualizate_graphs import create_bar_plot
from evaluation.evaluation_metrics import single_eval, get_ndcg, get_rsc

# Main evaluation script
def main():
    # Configuration and file paths
    model_path = "models/weights/model_graphsage.pth"
    data_dir = "data/"
    save_path = "results/metrics_bar_plot.png"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    data = DataManager(data_dir)

    # Initialize the model
    num_playlists = data.num_playlists()
    num_songs = data.num_songs()
    model = GraphSAGELinkPrediction(
        num_playlists,
        num_songs,
        playlist_dim=64,
        song_dim=64,
        dropout_prob=0.5,
    ).to(device)

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Generate outputs for training data
    with torch.no_grad():
        train_outputs = model(
            data.train_model_data()["train"]["playlist_ids"].to(device),
            data.train_model_data()["train"]["song_ids"].to(device),
        ).cpu().view(-1).numpy()

    # Generate outputs for test data
    with torch.no_grad():
        test_outputs = model(
            data.train_model_data()["test"]["playlist_ids"].to(device),
            data.train_model_data()["test"]["song_ids"].to(device),
        ).cpu().view(-1).numpy()

    # Create seeds
    train_seed = np.where(data.train_model_data()["train"]["label_ids"].numpy() == 1)[0]
    test_seed = np.where(data.train_model_data()["test"]["label_ids"].numpy() == 1)[0]


    # Additional metadata for evaluation
    train_labels = data.train_model_data()["train"]["label_ids"].numpy()
    test_labels = data.train_model_data()["test"]["label_ids"].numpy()
    num_classes = len(np.unique(train_labels))

    # Evaluate metrics using single_eval
    train_r_precision, train_hr_by_cls, train_cls_dist = single_eval(
        train_outputs, train_seed, train_labels, train_labels, num_classes
    )
    test_r_precision, test_hr_by_cls, test_cls_dist = single_eval(
        test_outputs, test_seed, test_labels, test_labels, num_classes
    )

    # Compute NDCG and RSC for train and test
    train_ndcg = get_ndcg(train_labels, train_outputs.argsort()[::-1])
    train_rsc = get_rsc(train_labels, train_outputs.argsort()[::-1])

    test_ndcg = get_ndcg(test_labels, test_outputs.argsort()[::-1])
    test_rsc = get_rsc(test_labels, test_outputs.argsort()[::-1])

    # Prepare metrics for visualization
    train_metrics = {
        "r_precision": train_r_precision,
        "ndcg": train_ndcg,
        "rsc": train_rsc,
    }
    test_metrics = {
        "r_precision": test_r_precision,
        "ndcg": test_ndcg,
        "rsc": test_rsc,
    }

    # Create bar plot
    create_bar_plot(train_metrics, test_metrics, save_path)


if __name__ == "__main__":
    main()