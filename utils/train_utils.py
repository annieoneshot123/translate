import torch
import torch.nn as nn
import torch.optim as optim

def train_epoch(model, data, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    for src, tgt in data:
        src = torch.tensor(src, dtype=torch.long).unsqueeze(0).to(device)
        tgt = torch.tensor(tgt, dtype=torch.long).unsqueeze(0).to(device)
        optimizer.zero_grad()
        output = model(src, tgt)
        # output: [batch_size, tgt_len, tgt_vocab_size]
        output_dim = output.shape[-1]
        output = output[:,1:].reshape(-1, output_dim)
        tgt = tgt[:,1:].reshape(-1)
        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data)

def evaluate(model, data, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, tgt in data:
            src = torch.tensor(src, dtype=torch.long).unsqueeze(0).to(device)
            tgt = torch.tensor(tgt, dtype=torch.long).unsqueeze(0).to(device)
            output = model(src, tgt, teacher_forcing_ratio=0.0)
            output_dim = output.shape[-1]
            output = output[:,1:].reshape(-1, output_dim)
            tgt = tgt[:,1:].reshape(-1)
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    return epoch_loss / len(data)
