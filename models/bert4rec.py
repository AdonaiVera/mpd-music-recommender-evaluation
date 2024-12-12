from transformers import BertConfig, BertTokenizer, BertForMaskedLM
import torch

class BERT4Rec:
    def __init__(self, vocab_size, max_seq_length, embedding_size=64, num_hidden_layers=3, num_attention_heads=2):
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=embedding_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_seq_length,
            type_vocab_size=1,
        )
        self.model = BertForMaskedLM(config)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
