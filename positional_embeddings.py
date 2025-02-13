import torch
class PositionalEncoding():
    def __init__(self, d_model, max_seq_length):
        self.d_model = d_model #dimenisons of noral embeddings
        self.max_seq_length = max_seq_length #max input
        self.pe = self.create_positional_encoding()

    def create_positional_encoding(self):
        """Create positional encoding matrix using PyTorch."""
        pe = torch.zeros(self.max_seq_length, self.d_model)  # Use PyTorch tensor

        # Position array (0, 1, 2, ..., max_seq_length - 1)
        position = torch.arange(self.max_seq_length).unsqueeze(1)  # Shape: (max_seq_length, 1)

        # Division term for the formula
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-torch.log(torch.tensor(10000.0)) / self.d_model))

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        return pe  # Returns a PyTorch tensor


    def __call__(self, x):
        """
        Add positional encoding to the input embeddings.

        Args:
            x (np.array): Input embeddings of shape (batch_size, seq_length, d_model).

        Returns:
            np.array: Input embeddings with positional encoding added.
        """
        # Get the actual sequence length from input x
        seq_length = x.size(1)  # Changed from x.size(0) to x.size(1) to get seq_length
        # Return the embeddings with positional encoding added
        return x + self.pe[:seq_length, :].unsqueeze(0) # unsqueeze to match embedding shape