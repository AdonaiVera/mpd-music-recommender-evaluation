# Music Recommendation System

This project is part of a Deep Learning class, it includes data preprocessing, exploratory data analysis (EDA), and recommendation techniques. 

## Table of Contents

1. [Getting Started](#getting-started)
2. [Project Structure](#project-structure)
3. [Preprocessing](#preprocessing)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
5. [Future Work](#future-work)
6. [Troubleshooting](#troubleshooting)

---

## Getting Started

To get started, clone this repository and install the necessary dependencies. The dataset will be downloaded and preprocessed using `pre-process.py`.

### Prerequisites

- Python 3.9
- Dependencies listed in `requirements.txt`

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AdonaiVera/mpd-music-recommender-evaluation
   cd mpd-music-recommender-evaluation
   ```

2. **Install dependencies:**
   ```bash
   # If you are going to run the graph code, you should use python3.9
   pip install -r requirements.txt
   ```

## Project Structure

Here's an overview of the project structure:

```
.
├── data/
│   ├── raw/               # Raw dataset downloaded here
│   ├── processed/         # Preprocessed files saved here
│       ├── tracks_df.csv      # Track data
│       ├── playlists_df.csv   # Playlist data
│       ├── playlist_tracks_df.csv # Playlist-track relationships
│
├── methods/
│   ├── pre-process.py     # Script to download and preprocess data
│   ├── exploration_data_analysis.py        # Functions for data analysis
│
├── README.md              # Project documentation
└── requirements.txt       # Required libraries
```

## Preprocessing

To download, preprocess, and save the dataset, run the following command:

```bash
python methods/pre_processing.py
```

### Steps

1. **Download the Dataset**:
   The script downloads the Spotify playlist dataset from a shared storage link provided in `pre-process.py`.

2. **Preprocess the Data**:
   Basic preprocessing includes parsing JSON files, validating and cleaning data, and generating three main files:
   - `tracks_df.csv`: Contains track information.
   - `playlists_df.csv`: Contains playlist metadata.
   - `playlist_tracks_df.csv`: Links playlists and tracks for recommendation modeling.

3. **Output**:
   The preprocessed files are saved in the `data/processed` directory.

   After running the preprocessing, the following files will be available in `data/processed`:
   - `tracks_df.csv`
   - `playlists_df.csv`
   - `playlist_tracks_df.csv`

## Exploratory Data Analysis (EDA) (Pending to build)

While basic preprocessing has been done, the next recommended step is to perform EDA to better understand the dataset. Here are some potential EDA tasks:

1. **Analyze Most Popular Tracks, Artists, and Albums**:
   - Identify the most frequently appearing tracks, artists, and albums.

2. **Playlist Diversity Analysis**:
   - Explore playlist characteristics such as artist diversity, duration distribution, and track counts.

3. **Correlation Analysis**:
   - Study correlations between playlist features, such as the number of tracks, albums, followers, and duration.

4. **Visualization**:
   - Generate plots to visualize popular tracks, artists, one-hit wonders, and duration distributions.

   *Tip:* You can create these plots using functions in the `plots.py` file and extend the analysis in `analysis.py`.


## Graphs Information
GraphSAGE-based recommendation model to predict playlist-song relevance by learning embeddings for playlists and songs, optimizing performance using weighted loss, and tracking metrics like AUC, precision, recall, and F1-score.


GraphSAGE (Graph Sample and Aggregate): 
Keys: 
1. Instead of using the entire graph, GraphSAGE samples a fixed number of neighbors for each node at each layer.
2. For each node, GraphSAGE aggregates features from its sampled neighbors
3. Recursive Neighborhood Expansion

Steps:
1. GraphSAGE looks at the nodes it's connected to (its neighbors, like songs in the playlist).
2. GraphSAGE combines information from all the connected neighbors. For example, for a playlist, it looks at all its songs and "summarizes" their features.
3. GraphSAGE mixes this summary with the original node's features (like the playlist name). This creates a new, richer version of the playlist's information.
4. GraphSAGE repeats this process for a few layers, so the node's updated information now includes details from neighbors, neighbors' neighbors, and so on.
5. The model has a learned embedding (a set of numbers) that captures all the relevant connections and features from the graph.
6. These embeddings are used to predict relationship

### Small version: 
Train, validation, and test splits created.
Total Songs: 122902
Total Playlists: 6000
Total Edges: 394387

Nodes: Songs, playlist
Edges: Representing user interactions as graph connections.

## Future Work

Building magic with the models ...

## Contributors

- **Alhim Adonai Vera Gonzalez** - [veragoaa@mail.uc.edu](mailto:veragoaa@mail.uc.edu)
- **Janire Pampin Rubio** - [pampinje@mail.uc.edu](mailto:pampinje@mail.uc.edu)
- **Leire Santamaria Lopez** - [santamle@mail.uc.edu](mailto:santamle@mail.uc.edu)
- **Jorge Gutiérrez Ubierna** - [gutierj2@mail.uc.edu](mailto:gutierj2@mail.uc.edu)
- **Daniel Vennemeyer** - [vennemdp@mail.uc.edu](mailto:vennemdp@mail.uc.edu)
- **Aneesh Deshmukh** - [deshmua2@mail.uc.edu](mailto:deshmua2@mail.uc.edu)
- **Autri Ilesh Banerjee** - [banerja2@mail.uc.edu](mailto:banerja2@mail.uc.edu)
- **Kai Liao** - [liaok@mail.uc.edu](mailto:liaok@mail.uc.edu)
