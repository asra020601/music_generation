import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from positional_embeddings import PositionalEncoding

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.masked_attention = MultiHeadAttention(d_model, num_heads)
        self.encoder_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Masked multi-head attention (self-attention on target sequence)
        attn_output = self.masked_attention(x, x, x, tgt_mask)
        x = x + self.dropout(attn_output)  # Residual connection
        x = self.norm1(x)  # Layer normalization

        # Encoder-decoder attention (attend to encoder output)
        attn_output = self.encoder_attention(x, encoder_output, encoder_output, src_mask)
        x = x + self.dropout(attn_output)  # Residual connection
        x = self.norm2(x)  # Layer normalization

        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)  # Residual connection
        x = self.norm3(x)  # Layer normalization

        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)  # Output layer

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        seq_length = x.size(1)

        # Add embeddings and positional encoding
        x = self.embeddings(x)  # Shape: (batch_size, seq_len, d_model)
        x = x + self.positional_encoding.pe[:seq_length, :]  # Add positional encoding
        x = self.dropout(x)

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        # Final linear layer to predict output tokens
        output = self.fc_out(x)  # Shape: (batch_size, seq_len, vocab_size)
        return output
