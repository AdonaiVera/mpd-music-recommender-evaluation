import os
import json
import pandas as pd
import torch

class DataManager:
    def __init__(self, cur_dir):
        self.cur_dir = cur_dir
        self.data_dir = os.path.join(cur_dir, "processed_graphs")
        self.info_path = os.path.join(self.data_dir, "info.json")

    def get_info(self):
        with open(self.info_path, "r") as f:
            return json.load(f)
        
    def num_playlists(self):
        df = self.csv('playlists')
        all_playlist_ids = set(df['playlist_id'])
        return len(all_playlist_ids)
    
    def num_songs(self):
        df = self.csv('songs')
        all_songs_ids = set(df['song_id'])
        return len(all_songs_ids)
        
    def songs_in_playlist(self, pid):
        df = self.csv('edges')
        songs = set(df[df['playlist_id'] == pid]['song_id'])
        return songs
    
    def train_model_data(self): 
        data = Data(self.cur_dir)
        train_playlist_ids, train_song_ids, train_label_ids = data.get.edge_data_parsed('train_edges')
        test_playlist_ids, test_song_ids, test_label_ids = data.get.edge_data_parsed('test_edges')
        val_playlist_ids, val_song_ids, val_label_ids = data.get.edge_data_parsed('val_edges')

        parsed_data = {
            'train': {
                'playlist_ids': train_playlist_ids,
                'song_ids': train_song_ids,
                'label_ids': train_label_ids,
            },
            'test': {
                'playlist_ids': test_playlist_ids,
                'song_ids': test_song_ids,
                'label_ids': test_label_ids,
            },
            'val': {
                'playlist_ids': val_playlist_ids,
                'song_ids': val_song_ids,
                'label_ids': val_label_ids,
            }
        }

        return parsed_data

    def save_info(self, info):
        with open(self.info_path, "w") as f:
            json.dump(info, f, indent=4)

    def load_csv(self, file_name):
        return pd.read_csv(os.path.join(self.data_dir, file_name))

    def save_csv(self, df, file_name):
        df.to_csv(os.path.join(self.data_dir, file_name), index=False)
