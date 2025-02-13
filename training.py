from transformer import Transformer
from dataloader import dataloader
from data import tokenizer
import torch
import torch.nn as nn

vocab_size = tokenizer.vocab_size
d_model = 512
num_epochs = 10
learning_rate = 0.0001

# Initialize components
model = Transformer(vocab_size=vocab_size, d_model=d_model)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_input, batch_target in dataloader:
        # Prepare masks
        tgt_mask = model.generate_mask(batch_target.size(1) - 1)  # -1 for shifted output
        src_padding_mask = (batch_input == 0)

        # Forward pass
        optimizer.zero_grad()
        output = model(
            src=batch_input,
            tgt=batch_target[:, :-1],  # Exclude last token
            tgt_mask=tgt_mask,
            src_padding_mask=src_padding_mask
        )

        # Compute loss
        loss = criterion(
            output.view(-1, vocab_size),
            batch_target[:, 1:].reshape(-1)  # Exclude first token
        )

        # Backprop
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")