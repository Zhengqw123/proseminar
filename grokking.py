#!/usr/bin/env python
# coding: utf-8




import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import math


## Use GPU if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



p = 97  # prime number
train_frac = 0.6  # training fraction
d_model = 128  # model dimension
n_layers = 2  # number of Transformer layers
n_heads = 4  # number of attention heads
batch_size = 512
base_lr = 0.001
warmup_updates = 10 # learning rate warm-up
update_count = 0
weight_decay = 1  # AdamW weight decay
max_steps = 50000



def modulo_operation(x: int, y: int, p: int) -> int:
    """
    Performs a specialized modular operation based on the parity of y:
    - When y is odd: Computes x divided by y modulo p (x * y^(-1) mod p)
    - When y is even: Computes (x² - y²) modulo p
    
    Parameters:
        x (int): First operand
        y (int): Second operand (determines operation type)
        p (int): Modulus (must be positive integer)
    
    Returns:
        int: Result of the modular operation
    
    Raises:
        ValueError: When y is odd and has no modular inverse modulo p
    """
    if y % 2 == 1:  # y is odd
        ## Helper function to compute modular inverse
        def mod_inverse(a: int, m: int) -> int:
            """Computes the modular inverse a⁻¹ mod m using extended Euclidean algorithm"""
            ## Extended Euclidean Algorithm implementation
            def extended_gcd(a: int, b: int) -> tuple:
                """Returns (gcd, x, y) such that a*x + b*y = gcd(a,b)"""
                if a == 0:
                    return b, 0, 1
                ## Recursively compute GCD and coefficients
                gcd, x1, y1 = extended_gcd(b % a, a)
                ## Update coefficients using recurrence relations
                x = y1 - (b // a) * x1
                y = x1
                return gcd, x, y
            
            ## Compute GCD and coefficient x
            gcd, x, _ = extended_gcd(a, m)
            if gcd != 1:
                raise ValueError("Modular inverse does not exist")
            return (x % m + m) % m # normalize inverse to positive value in [0, m-1]
        
        ## Compute y's modular inverse modulo p
        y_inv = mod_inverse(y, p)
        # Return (x * y⁻¹) mod p
        return (x * y_inv) % p
        
    else:  ## y is even
        return (x**2 - y**2) % p # compute (x² - y²) mod p





def generate_dataset(p, train_frac):
     """
    Generate all possible (a, b, a◦b) triplets modulo p, then split into training and validation.
    """
    pairs = [(a, b, modulo_operation(a,b,p)) for a in range(p) for b in range(p)] # x ◦ y 
    np.random.shuffle(pairs)
    split = int(len(pairs) * train_frac)
    train_data, val_data = pairs[:split], pairs[split:]
    return train_data, val_data

## Define the vocabulary and encoding: tokens are <a><op><b>=<c>
vocab = {'<a>': 0, '<b>': 1, '<op>': 2, '=': 3, **{f'<{i}>': i+4 for i in range(p)}}
vocab_size = len(vocab)





def encode(a, b, c):
    """
    Encode the operation a ◦ b = c as token IDs.
    """
    return [
        vocab[f'<{a}>'], vocab['<op>'], vocab[f'<{b}>'], vocab['='], vocab[f'<{c}>']
    ]

## Generate the dataset
train_data, val_data = generate_dataset(p, train_frac)
train_sequences = [encode(a, b, c) for a, b, c in train_data]
val_sequences = [encode(a, b, c) for a, b, c in val_data]

def to_tensor(sequences):
    """
    Convert list of token ID sequences to TensorDataset.
    """
    inputs = torch.tensor([seq[:-1] for seq in sequences], dtype=torch.long) # take first (n-1) elements from each sequence
    targets = torch.tensor([seq[-1] for seq in sequences], dtype=torch.long) # take the last element from each sequence
    return TensorDataset(inputs, targets)

## Apply encoding and convert to Datasets and DataLoaders
train_dataset = to_tensor(train_sequences)
val_dataset = to_tensor(val_sequences)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)




## Define a Transformer Model
class GrokkingTransformer(nn.Module):
    """
    Simple Transformer Encoder model for classification over token sequences.
    """
    def __init__(self):
        super().__init__()
        self.noise_std = 0.01

        ## Positional Encoding (fixed, not learned)
        self.embed = nn.Embedding(vocab_size, d_model)
        position = torch.arange(0, 4).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_embed = torch.zeros(4, d_model)
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        self.pos_embed = nn.Parameter(pos_embed, requires_grad=False)  
        
        ## Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*4,
            batch_first=True )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, vocab_size) # final linear projection
        
    def forward(self, x):
        """
        Forward pass of the model with causal masking and optional noise.
        """
        x = self.embed(x) + self.pos_embed  # token embedding plus positional encoding
        
        if self.training:  # add noise in training step
            x = x + torch.randn_like(x) * self.noise_std
            
        mask = torch.triu(torch.ones(4, 4, dtype=torch.bool,device=x.device), diagonal=1)
        x = self.transformer(x, mask=mask)
        x = x[:, -1, :]
        return self.fc(x) # classification output



model = GrokkingTransformer().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0,betas=[0.9, 0.98], weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

update_count = 0
## training loop (monitoring training/validation accuracy)
train_accs, val_accs = [], []

for step in range(max_steps):
    model.train()
    total_loss = 0

    ## Warm-up learning rate scheduling
    if update_count < warmup_updates:
        lr = base_lr * (update_count + 1) / warmup_updates
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    ## One training epoch
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        update_count += 1
        total_loss += loss.item()
    
    ## Evaluate every 100 steps
    if step % 100 == 0:
        model.eval()
        train_correct, train_total = 0, 0
        with torch.no_grad():
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                preds = logits.argmax(dim=1)
                train_correct += (preds == targets).sum().item()
                train_total += targets.size(0)
            train_acc = train_correct / train_total
        
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                preds = logits.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
            val_acc = val_correct / val_total
        
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        avg_loss = total_loss / len(train_loader)
        print(f"Step {step:6d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

## Plot and save training/validation accuracy over time
plt.plot(train_accs, label='Train')
plt.plot(val_accs, label='Validation')
plt.xlabel('Steps (x100)')
plt.ylabel('Accuracy')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.savefig('training_curve.png')
plt.show()

