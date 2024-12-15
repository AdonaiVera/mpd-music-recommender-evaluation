import torch
import numpy as np
from methods.data_utils import DataManager
from evaluation.visualizate_graphs import plot_graph

def main():
    # Configuration and file paths
    data_dir = "data/"
    graph_save_path = "results/train_graph.png"
    sample_size = 100  # Number of edges to sample (adjust as needed)

    # Initialize data manager
    data = DataManager(data_dir)

    # Load training data
    train_edges = data.csv("train_edges")

    if train_edges.empty:
        print("No edges found in train_edges.csv. Cannot plot the graph.")
        return

    # Plot and save the graph with edge sampling
    plot_graph(
        train_edges,
        save_path=graph_save_path,
        title="Training Graph Visualization",
        sample_size=sample_size  # Specify the number of edges to sample
    )
    print(f"Graph visualization saved to {graph_save_path}")

if __name__ == "__main__":
    main()
