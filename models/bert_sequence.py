import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Parameters
MAX_SEQ_LENGTH = 100  # Maximum sequence length for BERT4Rec
MASK_PROB = 0.15  # Probability of masking items

# Load datasets
playlists_df = pd.read_csv('processed/playlists_df.csv')
tracks_df = pd.read_csv('processed/tracks_df.csv')

# Extract sequences from playlists
sequences = playlists_df['tracks'].apply(eval).tolist()  # Convert string to list of track IDs

# Pad sequences to a uniform length
def pad_sequence(seq, max_length, pad_token=0):
    return seq[:max_length] + [pad_token] * max(0, max_length - len(seq))

padded_sequences = [pad_sequence(seq, MAX_SEQ_LENGTH) for seq in sequences]

# Mask some items in the sequences for training
def mask_sequence(sequence, mask_prob=MASK_PROB, mask_token=-1):
    masked_seq = []
    labels = []
    for item in sequence:
        if np.random.rand() < mask_prob:
            masked_seq.append(mask_token)  # Replace with mask token
            labels.append(item)  # Keep the original value as the label
        else:
            masked_seq.append(item)
            labels.append(0)  # Use 0 for unmasked items
    return masked_seq, labels

masked_sequences = []
labels = []

for seq in padded_sequences:
    masked_seq, label = mask_sequence(seq)
    masked_sequences.append(masked_seq)
    labels.append(label)

# Convert to numpy arrays
masked_sequences = np.array(masked_sequences)
labels = np.array(labels)

# Split into train/test sets
train_sequences, test_sequences, train_labels, test_labels = train_test_split(
    masked_sequences, labels, test_size=0.2, random_state=42
)

# Save the preprocessed data
np.save('train_sequences.npy', train_sequences)
np.save('test_sequences.npy', test_sequences)
np.save('train_labels.npy', train_labels)
np.save('test_labels.npy', test_labels)

print("Data preprocessing complete.")
print(f"Train sequences shape: {train_sequences.shape}")
print(f"Test sequences shape: {test_sequences.shape}")
