# -*- coding: utf-8 -*-
"""
Text Generation Example
Natural Language Processing

This script implements a language model for text generation using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Reproducibility configuration
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence):
        embedded = self.dropout(self.embedding(sequence))
        output, (hidden, cell) = self.lstm(embedded)
        predictions = self.fc(output)
        return predictions

    def generate(self, initial_sequence, max_length=100, temperature=1.0):
        self.eval()
        current_sequence = initial_sequence
        generated = []
        
        with torch.no_grad():
            for _ in range(max_length):
                predictions = self(current_sequence)
                predictions = predictions[-1, :] / temperature
                probs = torch.softmax(predictions, dim=0)
                next_token = torch.multinomial(probs, 1)
                generated.append(next_token.item())
                current_sequence = torch.cat([current_sequence[1:], next_token.unsqueeze(0)])
        
        return generated

def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    
    for batch in iterator:
        optimizer.zero_grad()
        
        text, targets = batch
        text, targets = text.to(device), targets.to(device)
        
        predictions = model(text)
        predictions = predictions.view(-1, predictions.shape[-1])
        targets = targets.view(-1)
        
        loss = criterion(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in iterator:
            text, targets = batch
            text, targets = text.to(device), targets.to(device)
            
            predictions = model(text)
            predictions = predictions.view(-1, predictions.shape[-1])
            targets = targets.view(-1)
            
            loss = criterion(predictions, targets)
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

if __name__ == "__main__":
    # Hiperparámetros
    VOCAB_SIZE = 20000
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    N_LAYERS = 2
    DROPOUT = 0.5
    
    # Inicialización del modelo
    model = LanguageModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Entrenamiento
    N_EPOCHS = 10
    best_valid_loss = float('inf')
    
    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_iterator, optimizer, criterion)
        valid_loss = evaluate(model, valid_iterator, criterion)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best-lm-model.pt')
        
        print(f'Epoch: {epoch+1}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tValid Loss: {valid_loss:.3f}')
    
    # Ejemplo de generación de texto
    initial_text = torch.tensor([1, 2, 3]).to(device)  # tokens iniciales
    generated_text = model.generate(initial_text, max_length=100, temperature=0.7)
    
    # Main training functions would follow... 