# -*- coding: utf-8 -*-
"""
Word Embeddings Example
Natural Language Processing

This script implements Word2Vec (Skip-gram model) using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import random

# Reproducibility configuration
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        output = self.output(embeds)
        return output

class Word2VecDataset(Dataset):
    def __init__(self, text, window_size=2):
        self.text = text
        self.window_size = window_size
        self.word_pairs = self._create_pairs()

    def _create_pairs(self):
        pairs = []
        for i, word in enumerate(self.text):
            window_start = max(0, i - self.window_size)
            window_end = min(len(self.text), i + self.window_size + 1)
            for j in range(window_start, window_end):
                if i != j:
                    pairs.append((word, self.text[j]))
        return pairs

    def __len__(self):
        return len(self.word_pairs)

    def __getitem__(self, idx):
        return self.word_pairs[idx]

def train_step(batch, model, optimizer, criterion):
    model.train()
    target_words, context_words = batch
    target_words = target_words.to(device)
    context_words = context_words.to(device)
    
    optimizer.zero_grad()
    outputs = model(target_words)
    loss = criterion(outputs, context_words)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train(model, train_loader, optimizer, criterion, n_epochs):
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in train_loader:
            loss = train_step(batch, model, optimizer, criterion)
            total_loss += loss
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}')

if __name__ == "__main__":
    # Hiperparámetros
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 300
    BATCH_SIZE = 64
    N_EPOCHS = 5
    LEARNING_RATE = 0.001
    
    # Inicialización del modelo
    model = SkipGramModel(VOCAB_SIZE, EMBEDDING_DIM)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Crear dataset y dataloader
    text = ["your", "text", "data", "here"]  
    dataset = Word2VecDataset(text)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Entrenamiento
    train(model, train_loader, optimizer, criterion, N_EPOCHS)
    
    # Guardar el modelo
    torch.save(model.state_dict(), 'word2vec-model.pt')
    
    # Obtener embeddings
    embeddings = model.embeddings.weight.data.cpu().numpy()
    