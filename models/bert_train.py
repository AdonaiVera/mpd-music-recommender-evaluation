from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
import numpy as np
import random
from .bert4rec import *

class PlaylistDataset(Dataset):
    def __init__(self, sequences, max_seq_length, tokenizer, mask_prob=0.15):
        self.sequences = sequences
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_ids = sequence[:self.max_seq_length]
        labels = input_ids.copy()

        # Masking
        for i in range(len(input_ids)):
            if random.random() < self.mask_prob:
                input_ids[i] = self.tokenizer.mask_token_id

        # Padding
        padding_length = self.max_seq_length - len(input_ids)
        input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
        labels.extend([-100] * padding_length)

        attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in input_ids]
        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels)

# Load preprocessed data
train_sequences = np.load('data/sequences/train_sequences.npy', allow_pickle=True)
labels = np.load('data/sequences/train_labels.npy', allow_pickle=True)

# Get vocab size
vocab_size = max(max(seq) for seq in train_sequences) + 1  # Add 1 for zero-based indexing
max_seq_length = max(len(seq) for seq in train_sequences)

sequences = train_sequences.tolist()  # For the training set
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = PlaylistDataset(sequences, max_seq_length, tokenizer)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Train the model
def train_model(model, data_loader, epochs=5, lr=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, attention_mask, labels in data_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

model = BERT4Rec(vocab_size, max_seq_length).model
train_model(model, data_loader)
