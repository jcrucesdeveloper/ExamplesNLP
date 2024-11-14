"""
Sequence Labelling Example
Natural Language Processing

This script implements sequence labeling tasks using neural networks with PyTorch.
Demonstrates Named Entity Recognition (NER) and Part-of-Speech (POS) tagging.
The model uses BiLSTM architecture with CRF layer for sequence prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
from seqeval.metrics import f1_score, precision_score, recall_score
import time

# Configuración de reproducibilidad
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Configuración de dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Definición del dataset
class TaggingDataset(Dataset):
    def __init__(self, paths, lower=False, separator=" ", encoding="utf-8"):
        data = []
        for path in paths:
            with open(path, 'r', encoding=encoding) as file:
                text, tag = [], []
                for line in file:
                    line = line.strip()
                    if line == "":
                        data.append({'text': text, 'nertags': tag})
                        text, tag = [], []
                    else:
                        line_content = line.split(separator)
                        text.append(line_content[0].lower() if lower else line_content[0])
                        tag.append(line_content[1])
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return item["nertags"], item["text"]

# Funciones de preprocesamiento
def fit_vocab(data_iter):
    def update_counter(counter_obj):
        sorted_by_freq_tuples = sorted(counter_obj.items(), key=lambda x: x[1], reverse=True)
        return OrderedDict(sorted_by_freq_tuples)

    counter_1, counter_2 = Counter(), Counter()
    for _nertags, _text in data_iter:
        counter_1.update(_text)
        counter_2.update(_nertags)

    v1 = vocab(update_counter(counter_1), specials=['<PAD>', '<unk>'])
    v1.set_default_index(v1["<unk>"])
    v2 = vocab(update_counter(counter_2), specials=['<PAD>'])

    return lambda x: v1(x), lambda x: v2(x), v1, v2

def collate_batch(batch, nertags_pipeline, text_pipeline, device):
    nertags_list, text_list = [], []
    for _nertags, _text in batch:
        nertags_list.append(torch.tensor(nertags_pipeline(_nertags), dtype=torch.int64))
        text_list.append(torch.tensor(text_pipeline(_text), dtype=torch.int64))
    nertags_list = pad_sequence(nertags_list, batch_first=True).T
    text_list = pad_sequence(text_list, batch_first=True).T
    return nertags_list.to(device), text_list.to(device)

# Definición del modelo
class NER_RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        outputs, _ = self.lstm(embedded)
        predictions = self.fc(self.dropout(outputs))
        return predictions

# Funciones de entrenamiento y evaluación
def calculate_metrics(preds, y_true, pad_idx, o_idx):
    y_pred = preds.argmax(dim=1, keepdim=True)
    mask = [(y_true != pad_idx)]
    y_pred = y_pred[mask].view(-1).to('cpu').numpy()
    y_true = y_true[mask].to('cpu').numpy()
    y_pred = [[NER_TAGS.vocab.get_itos()[v] for v in y_pred]]
    y_true = [[NER_TAGS.vocab.get_itos()[v] for v in y_true]]
    f1 = f1_score(y_true, y_pred, mode='strict')
    precision = precision_score(y_true, y_pred, mode='strict')
    recall = recall_score(y_true, y_pred, mode='strict')
    return precision, recall, f1

def train(model, iterator, optimizer, criterion):
    epoch_loss, epoch_precision, epoch_recall, epoch_f1 = 0, 0, 0, 0
    model.train()
    for tags, text in iterator:
        optimizer.zero_grad()
        predictions = model(text.to(device))
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = torch.reshape(tags, (-1,)).to(device)
        loss = criterion(predictions, tags)
        precision, recall, f1 = calculate_metrics(predictions, tags, PAD_TAG_IDX, O_TAG_IDX)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_precision += precision
        epoch_recall += recall
        epoch_f1 += f1
    return epoch_loss / len(iterator), epoch_precision / len(iterator), epoch_recall / len(iterator), epoch_f1 / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss, epoch_precision, epoch_recall, epoch_f1 = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for tags, text in iterator:
            predictions = model(text.to(device))
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = torch.reshape(tags, (-1,)).to(device)
            loss = criterion(predictions, tags)
            precision, recall, f1 = calculate_metrics(predictions, tags, PAD_TAG_IDX, O_TAG_IDX)
            epoch_loss += loss.item()
            epoch_precision += precision
            epoch_recall += recall
            epoch_f1 += f1
    return epoch_loss / len(iterator), epoch_precision / len(iterator), epoch_recall / len(iterator), epoch_f1 / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    return int(elapsed_time / 60), int(elapsed_time - (elapsed_time / 60) * 60)

# Inicialización y configuración del modelo
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

# Main script
if __name__ == "__main__":
    # Carga de datos
    data_iter = TaggingDataset(["train.txt", "dev.txt"])
    data_length = len(data_iter)
    train_length = int(data_length * 0.8)
    dev_length = int(data_length * 0.1)
    test_length = data_length - train_length - dev_length
    train_iter, dev_iter, test_iter = torch.utils.data.random_split(data_iter, (train_length, dev_length, test_length), torch.Generator().manual_seed(42))
    text_pipeline, nertags_pipeline, TEXT, NER_TAGS = fit_vocab(train_iter)

    # Configuración de hiperparámetros
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    OUTPUT_DIM = len(NER_TAGS.vocab)
    N_LAYERS = 3
    DROPOUT = 0.6
    BIDIRECTIONAL = True
    BATCH_SIZE = 22

    # Índices especiales
    UNK_IDX = TEXT.vocab.get_stoi()['<unk>']
    PAD_IDX = TEXT.vocab.get_stoi()['<PAD>']
    PAD_TAG_IDX = NER_TAGS.get_stoi()['<PAD>']
    O_TAG_IDX = NER_TAGS.vocab.get_stoi()['O']

    # Preparación de dataloaders
    fix_collocate_batch = lambda x: collate_batch(x, nertags_pipeline, text_pipeline, device)
    dataloader_train = DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=False, collate_fn=fix_collocate_batch)
    dataloader_dev = DataLoader(dev_iter, batch_size=BATCH_SIZE, shuffle=False, collate_fn=fix_collocate_batch)
    dataloader_test = DataLoader(test_iter, batch_size=BATCH_SIZE, shuffle=False, collate_fn=fix_collocate_batch)

    # Inicialización del modelo
    model = NER_RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
    model.apply(init_weights)
    model = model.to(device)

    # Configuración de optimizador y criterio
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TAG_IDX).to(device)

    # Entrenamiento del modelo
    n_epochs = 10
    best_valid_loss = float('inf')
    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss, train_precision, train_recall, train_f1 = train(model, dataloader_train, optimizer, criterion)
        valid_loss, valid_precision, valid_recall, valid_f1 = evaluate(model, dataloader_dev, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pt')

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train f1: {train_f1:.2f} | Train precision: {train_precision:.2f} | Train recall: {train_recall:.2f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. f1: {valid_f1:.2f} |  Val. precision: {valid_precision:.2f} | Val. recall: {valid_recall:.2f}')

    # Evaluación del modelo final
    model.load_state_dict(torch.load('best_model.pt'))
    valid_loss, valid_precision, valid_recall, valid_f1 = evaluate(model, dataloader_dev, criterion)
    print(f'Val. Loss: {valid_loss:.3f} |  Val. f1: {valid_f1:.2f} | Val. precision: {valid_precision:.2f} | Val. recall: {valid_recall:.2f}')
    prueba_loss, prueba_precision, prueba_recall, prueba_f1 = evaluate(model, dataloader_test, criterion)
    print(f'Test Loss: {prueba_loss:.3f} | Test f1: {prueba_f1:.2f} | Test precision: {prueba_precision:.2f} | Test recall: {prueba_recall:.2f}')