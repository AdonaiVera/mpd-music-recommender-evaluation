import os
import pandas as pd
import json
import random
import math
import torch
import requests
from zipfile import ZipFile

class SpotifyGraphPreprocessor:
    def __init__(self, data_folder="data", dataset_url=None):
        self.data_folder = data_folder
        self.raw_folder = os.path.join(data_folder, "raw")
        self.raw_folder_data = os.path.join(self.raw_folder, "data")
        self.processed_folder = os.path.join(data_folder, "processed_graphs")
        self.dataset_url = dataset_url

    def download_and_extract_dataset(self):
        """
        Downloads and extracts the Spotify Million Playlist Dataset.
        """
        if not os.path.exists(self.raw_folder):
            os.makedirs(self.raw_folder)
            dataset_zip_path = os.path.join(self.raw_folder, "spotify_dataset.zip")
            if not os.path.exists(dataset_zip_path):
                print("Downloading dataset...")
                response = requests.get(self.dataset_url)
                with open(dataset_zip_path, 'wb') as f:
                    f.write(response.content)
            
            print("Extracting dataset...")
            with ZipFile(dataset_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_folder)
            print("Dataset downloaded and extracted.")

    def initialize_processed_data(self):
        """
        Creates empty processed files for songs, playlists, and edges if they don't exist.
        """
        if not os.path.exists(self.processed_folder):
            os.makedirs(self.processed_folder)
        
        song_file = os.path.join(self.processed_folder, "songs.csv")
        playlist_file = os.path.join(self.processed_folder, "playlists.csv")
        edge_file = os.path.join(self.processed_folder, "edges.csv")
        
        # Create empty files if they don't exist
        if not os.path.exists(song_file):
            pd.DataFrame(columns=["song_id", "track_uri", "track_name", "artist_name"]).to_csv(song_file, index=False)
        if not os.path.exists(playlist_file):
            pd.DataFrame(columns=["playlist_id", "playlist_name", "num_tracks", "num_followers"]).to_csv(playlist_file, index=False)
        if not os.path.exists(edge_file):
            pd.DataFrame(columns=["playlist_id", "song_id", "label"]).to_csv(edge_file, index=False)

    def process_playlist_data(self, small_version=False):
        """
        Processes raw Spotify playlist JSON files into structured song, playlist, and edge datasets.
        """
        print("Processing raw data into structured datasets...")
        song_data, playlist_data, edge_data = [], [], []
        song_id_map = {}
        song_id_counter = 0
        playlist_id_counter = 0
        limit = 5
        nCount = 0  

        for json_file in os.listdir(self.raw_folder_data):
            if json_file.startswith("mpd.slice.") and json_file.endswith(".json"):
                file_path = os.path.join(self.raw_folder_data, json_file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                for playlist in data['playlists']:
                    # Process playlist
                    playlist_data.append({
                        "playlist_id": playlist_id_counter,
                        "playlist_name": playlist["name"],
                        "num_tracks": playlist["num_tracks"],
                        "num_followers": playlist["num_followers"]
                    })

                    # Process tracks
                    playlist_song_ids = set()
                    for track in playlist["tracks"]:
                        track_uri = track["track_uri"]
                        if track_uri not in song_id_map:
                            song_id_map[track_uri] = song_id_counter
                            song_data.append({
                                "song_id": song_id_counter,
                                "track_uri": track_uri,
                                "track_name": track["track_name"],
                                "artist_name": track["artist_name"]
                            })
                            song_id_counter += 1
                        
                        # Add positive edge
                        playlist_song_ids.add(song_id_map[track_uri])
                        edge_data.append({
                            "playlist_id": playlist_id_counter,
                            "song_id": song_id_map[track_uri],
                            "label": 1  # Positive edge
                        })

                    # Add negative edges
                    all_song_ids = set(song_id_map.values())
                    negative_song_ids = all_song_ids - playlist_song_ids
                    for neg_song_id in random.sample(negative_song_ids, min(len(negative_song_ids), len(playlist_song_ids))):
                        edge_data.append({
                            "playlist_id": playlist_id_counter,
                            "song_id": neg_song_id,
                            "label": 0  # Negative edge
                        })

                    playlist_id_counter += 1

                if nCount == limit and small_version:
                    break
                
                nCount += 1

        # Save to processed files
        pd.DataFrame(song_data).to_csv(os.path.join(self.processed_folder, "songs.csv"), index=False)
        pd.DataFrame(playlist_data).to_csv(os.path.join(self.processed_folder, "playlists.csv"), index=False)
        pd.DataFrame(edge_data).to_csv(os.path.join(self.processed_folder, "edges.csv"), index=False)
        print("Processing complete!")


    def create_train_test_val_splits(self, splits=(0.7, 0.15, 0.15)):
        """
        Splits the edge data into training, validation, and test datasets.
        """
        edge_file = os.path.join(self.processed_folder, "edges.csv")
        edges = pd.read_csv(edge_file)
        edges = edges.sample(frac=1, random_state=42)  # Shuffle the data

        num_edges = len(edges)
        train_end = int(num_edges * splits[0])
        val_end = train_end + int(num_edges * splits[1])

        train_edges = edges[:train_end]
        val_edges = edges[train_end:val_end]
        test_edges = edges[val_end:]

        train_edges.to_csv(os.path.join(self.processed_folder, "train_edges.csv"), index=False)
        val_edges.to_csv(os.path.join(self.processed_folder, "val_edges.csv"), index=False)
        test_edges.to_csv(os.path.join(self.processed_folder, "test_edges.csv"), index=False)
        print("Train, validation, and test splits created.")

    def summarize_data(self):
        """
        Summarizes the processed dataset and saves it to info.json.
        """
        songs = pd.read_csv(os.path.join(self.processed_folder, "songs.csv"))
        playlists = pd.read_csv(os.path.join(self.processed_folder, "playlists.csv"))
        edges = pd.read_csv(os.path.join(self.processed_folder, "edges.csv"))

        # Compute summary
        total_songs = len(songs)
        total_playlists = len(playlists)
        total_edges = len(edges)

        # Split edges into train, validation, and test
        train_edges = pd.read_csv(os.path.join(self.processed_folder, "train_edges.csv"))
        val_edges = pd.read_csv(os.path.join(self.processed_folder, "val_edges.csv"))
        test_edges = pd.read_csv(os.path.join(self.processed_folder, "test_edges.csv"))

        pos_edges = len(edges[edges['label'] == 1])
        neg_edges = len(edges[edges['label'] == 0])

        summary = {
            "files_parsed": len(os.listdir(self.processed_folder)),
            "playlists": total_playlists,
            "songs": total_songs,
            "train_edges": len(train_edges),
            "val_edges": len(val_edges),
            "test_edges": len(test_edges),
            "pos_edges": pos_edges,
            "neg_edges": neg_edges,
        }

        # Save to info.json
        info_path = os.path.join(self.processed_folder, "info.json")
        with open(info_path, "w") as json_file:
            json.dump(summary, json_file, indent=4)

        # Print summary
        print(f"Summary saved to {info_path}")
        print(json.dumps(summary, indent=4))


    def pipeline(self, DATA_FOLDER, DATASET_URL):
        # Running the full pipeline as a function in case.
        preprocessor = SpotifyGraphPreprocessor(DATA_FOLDER, DATASET_URL)
        preprocessor.download_and_extract_dataset()
        preprocessor.initialize_processed_data()
        preprocessor.process_playlist_data()
        preprocessor.create_train_test_val_splits()
        preprocessor.summarize_data()



if __name__ == "__main__":
    DATA_FOLDER = "data"
    DATASET_URL = "https://storage.googleapis.com/tecla/spotify-million-playlist-dataset/spotify_million_playlist_dataset.zip"
    
    # If you want to have small version of dataset set true
    small_version = True

    preprocessor = SpotifyGraphPreprocessor(DATA_FOLDER, DATASET_URL)
    preprocessor.download_and_extract_dataset()
    preprocessor.initialize_processed_data()
    preprocessor.process_playlist_data(small_version)
    preprocessor.create_train_test_val_splits()
    preprocessor.summarize_data()
