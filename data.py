
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict
import tiktoken
# Replace 'your_file.csv' with the actual name of your file
# and adjust the path if necessary
df = pd.read_csv('dataset/english_cleaned_lyrics.csv')
print(df.head())

tokenizer = tiktoken.get_encoding("gpt2")
class LyricsDataset(Dataset):
    def __init__(
        self,
        lyrics: List[str],          # List of lyrics from df['lyrics']
        tokenizer,  # Tokenizer (word-to-index dictionary)
        window_size: int,           # Input sequence length
        shift: int = 1,             # Slide window by `shift` tokens
        pad_token: int = tokenizer.eot_token     # Padding token ID
    ):
        self.lyrics = lyrics
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.shift = shift
        self.pad_token = pad_token
        self.pairs = self._generate_pairs()

    def _tokenize(self, text: str) -> List[int]:
        """Convert text to token IDs using the tokenizer."""
        return tokenizer.encode(text)

    def _pad_sequence(self, sequence: List[int], max_len: int) -> List[int]:
        """Pad or truncate a sequence to `max_len`."""
        if len(sequence) >= max_len:
            return sequence[:max_len]
        return sequence + [self.pad_token] * (max_len - len(sequence))

    def _generate_pairs(self) -> List[tuple]:
        """Generate input-target pairs with sliding window and padding."""
        pairs = []
        for lyric in self.lyrics:
            tokens = self._tokenize(lyric)
            # Slide over tokens to create input-target pairs
            for i in range(0, len(tokens) - self.window_size + 1, self.shift):
                input_seq = tokens[i:i + self.window_size]
                target_seq = tokens[i + 1:i + self.window_size + 1]
                # Pad sequences to ensure fixed length
                input_seq = self._pad_sequence(input_seq, self.window_size)
                target_seq = self._pad_sequence(target_seq, self.window_size)
                pairs.append((input_seq, target_seq))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple:
        input_seq, target_seq = self.pairs[idx]
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long)
        )