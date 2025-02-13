import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention:
    def __init__(self, query, key, value):
        """
        Initialize the Attention mechanism.

        Args:
            query (torch.Tensor): Query tensor of shape (seq_len, d_k).
            key (torch.Tensor): Key tensor of shape (seq_len, d_k).
            value (torch.Tensor): Value tensor of shape (seq_len, d_v).
        """
        self.query = query
        self.key = key
        self.value = value
        self.scores = None
        self.attention = None

    def forward(self):
        """
        Compute the attention output.

        Returns:
            torch.Tensor: Attention output of shape (seq_len, d_v).
            torch.Tensor: Attention weights of shape (seq_len, seq_len).
        """
        # Step 1: Compute attention scores (scaled dot-product)
        d_k = self.key.size(-1)  # Dimension of key vectors
        self.scores = torch.matmul(self.query, self.key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))

        # Step 2: Apply softmax to get attention weights
        self.weights = F.softmax(self.scores, dim=-1)

        # Step 3: Compute attention output
        self.attention = torch.matmul(self.weights, self.value)

        return self.attention, self.weights
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear transformations
        Q = self.query(query)  # Shape: (batch_size, seq_len, d_model)
        K = self.key(key)      # Shape: (batch_size, seq_len, d_model)
        V = self.value(value)  # Shape: (batch_size, seq_len, d_model)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e20"))
        weights = F.softmax(scores, dim=-1)
        attention = torch.matmul(weights, V)

        # Concatenate heads
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear layer
        output = self.fc_out(attention)
        return output