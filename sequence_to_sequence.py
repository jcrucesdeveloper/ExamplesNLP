# This file implements several neural network models for natural language processing tasks:
# • Sequence-to-sequence translation model with attention mechanism for Spanish-English translation
# • BERT-based model for next sentence prediction to determine the most probable following sentence
# • BERT-based masked language model for predicting masked words in sentences
# • BERT-based question answering model for extracting answers from context

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import math
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from transformers import BertTokenizer, BertForNextSentencePrediction, BertForMaskedLM, BertForQuestionAnswering

# Constants
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 5
SEED = 1234
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "*", 1: "STOP"}
        self.n_tokens = 2

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_tokens
            self.word2count[word] = 1
            self.index2word[self.n_tokens] = word
            self.n_tokens += 1
        else:
            self.word2count[word] += 1

def normalize_string(s):
    s = unicodedata.normalize('NFC', s)
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?áéíóúñ´]+", r" ", s)
    return s.strip()

def filter_pairs(pairs, max_length):
    return [p for p in pairs if len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length]

def read_dataset(path, reverse=False):
    lines = open(path, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalize_string(s) for s in l.split('\t')][:2] for l in lines]
    
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang("spa")
        output_lang = Lang("eng")
    else:
        input_lang = Lang("eng")
        output_lang = Lang("spa")
    
    return input_lang, output_lang, pairs

def read_langs(lang1, lang2, reverse=False, max_length=10):
    input_lang, output_lang, pairs = read_dataset("spa.txt", reverse)
    pairs = filter_pairs(pairs, max_length)
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    return input_lang, output_lang, pairs

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

class MyAttention(nn.Module):
    def __init__(self, hidden_size):
        super(MyAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = MyAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)
        return output, hidden, attn_weights

# Training functions
def sentence2indexes(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def sentence2tensor(lang, sentence):
    indexes = sentence2indexes(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def get_dataloader(batch_size):
    input_lang, output_lang, pairs = read_langs('eng', 'spa', reverse=True, max_length=5)
    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = sentence2indexes(input_lang, inp)
        tgt_ids = sentence2indexes(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                              torch.LongTensor(target_ids).to(device))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader

def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
        loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1))
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001, print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (Época %d, %d%% completado) Pérdida Promedio: %.4f' % (
                time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

# BERT functions
def oracion_mas_probable(first, sentA, sentB):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    
    encodingA = tokenizer(first, sentA, return_tensors='pt')
    encodingB = tokenizer(first, sentB, return_tensors='pt')
    
    outputA = model(**encodingA, labels=torch.LongTensor([1]))
    outputB = model(**encodingB, labels=torch.LongTensor([1]))
    
    logitsA = outputA.logits
    logitsB = outputB.logits
    
    probA = torch.softmax(logitsA, dim=1)[0][0].item()
    probB = torch.softmax(logitsB, dim=1)[0][0].item()
    
    return "La oración que continúa más probable es A" if probA > probB else "La oración que continúa más probable es B"

def palabra_mas_probable(sentence):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    
    tokenized_text = tokenizer.tokenize(sentence)
    masked_index = tokenized_text.index('[MASK]')
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    
    predictions = model(tokens_tensor).logits
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    
    return predicted_token

def entregar_respuesta(qst, cntxt):
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    
    inputs = tokenizer(qst, cntxt, return_tensors='pt')
    start_positions = torch.tensor([1])
    end_positions = torch.tensor([3])
    
    outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1
    
    all_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    answer = tokenizer.convert_tokens_to_string(all_tokens[start_index:end_index])
    
    return answer