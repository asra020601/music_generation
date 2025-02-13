from data import tokenizer, LyricsDataset, df

# Define parameters
window_size = 5  # Input sequence length
shift = 1        # Slide window by 1 token

# Create dataset
dataset = LyricsDataset(
    lyrics=df['lyrics'].tolist(),
    tokenizer= tokenizer,
    window_size=window_size,
    shift=shift,
    pad_token = tokenizer.eot_token
)