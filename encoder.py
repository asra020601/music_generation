import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from positional_embeddings import PositionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention
        attn_output = self.attention(x, x, x, mask)
        x = x + self.dropout(attn_output)  # Residual connection
        x = self.norm1(x)  # Layer normalization

        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)  # Residual connection
        x = self.norm2(x)  # Layer normalization

        return x
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        seq_length = x.size(1)

        # Add embeddings and positional encoding
        x = self.embeddings(x)  # Shape: (batch_size, seq_len, d_model)
        x = x + self.positional_encoding.pe[:seq_length, :]  # Add positional encoding
        x = self.dropout(x)

        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)

        return x