import os
import json
import pandas as pd
import torch


class DataManager:
    def __init__(self, cur_dir):
        """
        Initializes the DataManager with directories and file paths.
        """
        self.cur_dir = cur_dir
        self.data_dir = os.path.join(cur_dir, "processed_graphs")
        self.model_dir = "models/weights"
        self.info_path = os.path.join(self.data_dir, "info.json")

    # Get Methods
    def get_info(self):
        """
        Reads the info.json file to retrieve dataset summary information.
        """
        with open(self.info_path, "r") as f:
            return json.load(f)

    def csv(self, filename):
        """
        Reads a CSV file from the processed data directory.
        """
        file_path = os.path.join(self.data_dir, f"{filename}.csv")
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            print(f"{filename}.csv not found.")
            return pd.DataFrame()

    def num_playlists(self):
        """
        Counts the number of unique playlists.
        """
        playlists = self.csv("playlists")
        return playlists["playlist_id"].nunique()

    def num_songs(self):
        """
        Counts the number of unique songs.
        """
        songs = self.csv("songs")
        return songs["song_id"].nunique()

    def songs_in_playlist(self, playlist_id):
        """
        Retrieves all songs associated with a specific playlist.
        """
        edges = self.csv("edges")
        return set(edges[edges["playlist_id"] == playlist_id]["song_id"])

    def train_model_data(self):
        """
        Loads and prepares training, validation, and test data for the model.
        """
        train_edges = self.csv("train_edges")
        val_edges = self.csv("val_edges")
        test_edges = self.csv("test_edges")

        return {
            "train": {
                "playlist_ids": torch.LongTensor(train_edges["playlist_id"].values),
                "song_ids": torch.LongTensor(train_edges["song_id"].values),
                "label_ids": torch.LongTensor(train_edges["label"].values),
            },
            "val": {
                "playlist_ids": torch.LongTensor(val_edges["playlist_id"].values),
                "song_ids": torch.LongTensor(val_edges["song_id"].values),
                "label_ids": torch.LongTensor(val_edges["label"].values),
            },
            "test": {
                "playlist_ids": torch.LongTensor(test_edges["playlist_id"].values),
                "song_ids": torch.LongTensor(test_edges["song_id"].values),
                "label_ids": torch.LongTensor(test_edges["label"].values),
            },
        }

    def save_model_weights(self, model):
        """
        Saves the trained model weights.
        """
        torch.save(model.state_dict(), os.path.join(self.model_dir, "model_weights.pth"))

    def save_training_data(self, output_data):
        """
        Saves training metrics to a CSV file.
        """
        metrics_df = pd.DataFrame({
            "Epoch": list(range(1, len(output_data["train_losses"]) + 1)),
            "Loss": output_data["train_losses"],
            "Train AUC": output_data["train_auc_scores"],
            "Validation AUC": output_data["val_auc_scores"],
            "Test AUC": output_data["test_auc_scores"],
            "Precision": output_data["precision_scores"],
            "Recall": output_data["recall_scores"],
            "F1 Score": output_data["f1_scores"],
        })
        metrics_df.to_csv(os.path.join(self.model_dir, "training_metrics.csv"), index=False)

    # Update info.json
    def update_info(self, **kwargs):
        """
        Updates the info.json file with new values.
        """
        info = self.get_info()
        info.update(kwargs)
        with open(self.info_path, "w") as f:
            json.dump(info, f, indent=4)
