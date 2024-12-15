from .bert_train import *

def generate_predictions(model, sequences, max_seq_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for sequence in sequences:
            input_ids = sequence[:max_seq_length]
            input_ids += [tokenizer.pad_token_id] * (max_seq_length - len(input_ids))
            input_ids = torch.tensor([input_ids]).to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=-1).squeeze().tolist()
            predictions.append(pred_ids)
    return predictions

# Exampe: Make recommendations
test_sequences = np.load('data/sequences/test_sequences.npy', allow_pickle=True)
test_labels = np.load('data/sequences/test_labels.npy', allow_pickle=True)
test_sequences = test_sequences.tolist()  # For the test set

predictions = generate_predictions(model, test_sequences, max_seq_length)
print(predictions)
