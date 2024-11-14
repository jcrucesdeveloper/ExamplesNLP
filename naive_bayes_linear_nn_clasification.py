# This file implements and evaluates three different text classification models:
# 1. Multinomial Naive Bayes
# 2. Linear Model
# 3. Neural Network
# The models are trained and tested on a small dataset of sentences labeled as questions (?), 
# positive (+) or negative (-) statements.

import pandas as pd
from collections import namedtuple
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# Initialize dataset
document = namedtuple("document", ("words", "class_"))

raw_train_set = [
    ['Do you have plenty of time?', '?'],
    ['Does she have enough money?','?'],
    ['Did they have any useful advice?','?'],
    ['What day is today?','?'],
    ["I don't have much time",'-'],
    ["She doesn't have any money",'-'],
    ["They didn't have any advice to offer",'-'],
    ['Have you plenty of time?','?'],
    ['Has she enough money?','?'],
    ['Had they any useful advice?','?'],
    ["I haven't much time",'-'],
    ["She hasn't any money",'-'],
    ["He hadn't any advice to offer",'-'],
    ['How are you?','?'],
    ['How do you make questions in English?','?'],
    ['How long have you lived here?','?'],
    ['How often do you go to the cinema?','?'],
    ['How much is this dress?','?'],
    ['How old are you?','?'],
    ['How many people came to the meeting?','?'],
    ['I'm from France','+'],
    ['I come from the UK','+'],
    ['My phone number is 61709832145','+'],
    ['I work as a tour guide for a local tour company','+'],
    ['I'm not dating anyone','-'],
    ['I live with my wife and children','+'],
    ['I often do morning exercises at 6am','+'],
    ['I run everyday','+'],
    ['She walks very slowly','+'],
    ['They eat a lot of meat daily','+'],
    ['We were in France that day', '+'],
    ['He speaks very fast', '+'],
    ['They told us they came back early', '+'],
    ["I told her I'll be there", '+']
]

raw_test_set = [
    ['Do you know who lives here?','?'],
    ['What time is it?','?'],
    ['Can you tell me where she comes from?','?'],
    ['How are you?','?'],
    ['I fill good today', '+'],
    ['There is a lot of history here','+'],
    ['I love programming','+'],
    ['He told us not to make so much noise','+'],
    ['We were asked not to park in front of the house','+'],
    ["I don't have much time",'-'],
    ["She doesn't have any money",'-'],
    ["They didn't have any advice to offer",'-'],
    ['I am not really sure','+']
]

# Tokenize and create DataFrames
tokenized_train_set = [document(words=tuple(word_tokenize(d[0].lower())), class_=d[1]) for d in raw_train_set]
train_set = pd.DataFrame(data=tokenized_train_set)

tokenized_test_set = [document(words=tuple(word_tokenize(d[0].lower())), class_=d[1]) for d in raw_test_set]
test_set = pd.DataFrame(data=tokenized_test_set)

# Split into X and y
X_train, y_train = train_set.drop(columns="class_"), train_set["class_"]
X_test, y_test = test_set.drop(columns="class_"), test_set["class_"]

# Model 1: Multinomial Naive Bayes
class MyMultinomialNB():
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.vocab = set()
        self.clases = {}
        self.prob_c = {}
        self.prob_w_c = {}
        self.document_c = {}

    def fit(self, X, y):
        # Extract vocabulary
        for row in X.iterrows():
            doc_words = row[1]['words']
            for word in doc_words:
                self.vocab.add(word)
        self.vocab = list(self.vocab)

        # Calculate class probabilities
        for clase in y:
            if clase in self.clases:
                self.clases[clase] += 1
            else:
                self.clases[clase] = 1

        for clase in self.clases:
            self.prob_c[clase] = self.clases[clase] / len(X)

        # Calculate word probabilities given class
        for i, row in X.iterrows():
            w = list(row['words'])
            clase = y[i]
            if clase in self.document_c:
                self.document_c[clase] = self.document_c[clase] + w
            else:
                self.document_c[clase] = w

        for palabra in self.vocab:
            dict_palabra_clase = {}
            for clase in self.clases:
                mega_document = self.document_c[clase]
                num_w_in_c = mega_document.count(palabra)
                N = len(mega_document)
                p_alpha_numerador = num_w_in_c + self.alpha
                p_alpha_denominador = N + (self.alpha * len(self.vocab))
                p_alpha = p_alpha_numerador / p_alpha_denominador
                dict_palabra_clase[clase] = p_alpha
            self.prob_w_c[palabra] = dict_palabra_clase

    def log_sum_p_x_c(self, x, clase):
        sum = 0
        for word in x:
            if word in self.prob_w_c:
                if self.prob_w_c[word][clase] != 0:
                    sum += np.log10(self.prob_w_c[word][clase])
        return sum

    def predict(self, X):
        output_predicted = []
        for i, row in X.iterrows():
            documento = list(row['words'])
            document_prob_class = {}
            for clase in self.clases:
                p_cj = np.log10(self.prob_c[clase])
                sum_px_cj = self.log_sum_p_x_c(documento, clase)
                p = p_cj + sum_px_cj
                document_prob_class[clase] = p
            arg_max = max(document_prob_class, key=lambda k: document_prob_class[k])
            output_predicted.append(arg_max)
        return pd.Series(output_predicted)

# Model 2: Linear Model
class MyLinearModel():
    def __init__(self):
        self.count_vect = None
        self.label_encoder = None
        self.W = None
        self.b = None

    def fit(self, X, y, learning_rate, epochs, verbose=False):
        self.label_encoder = LabelEncoder()
        y_numeric = self.label_encoder.fit_transform(y)
        one_hot_matrix = np.zeros((len(y), len(self.label_encoder.classes_)))
        for i in range(len(y)):
            one_hot_matrix[i, y_numeric[i]] = 1

        X_list = []
        for row in X.iterrows():
            doc_words = row[1]['words']
            mi_lista = []
            for word in doc_words:
                mi_lista.append(word)
            X_list.append(mi_lista)

        self.count_vect = CountVectorizer()
        X_vec = self.count_vect.fit_transform([' '.join(words) for words in X_list])
        X_vec_dense = X_vec.toarray()

        n_classes = one_hot_matrix.shape[1]
        len_vocab = X_vec_dense.shape[1]
        self.W = np.random.rand(len_vocab, n_classes)
        self.b = np.random.rand(n_classes)

        for epoch in range(epochs):
            X_vec_dense, one_hot_matrix = shuffle(X_vec_dense, one_hot_matrix)
            for i in range(len(X_vec_dense)):
                x = X_vec_dense[i]
                y_true = one_hot_matrix[i]
                
                # Forward pass
                z = np.dot(x, self.W) + self.b
                y_pred = np.exp(z) / np.sum(np.exp(z))
                
                # Backward pass
                dz = y_pred - y_true
                dW = np.outer(x, dz)
                db = dz
                
                # Update parameters
                self.W -= learning_rate * dW
                self.b -= learning_rate * db

            if verbose:
                y_pred = self.predict(X)
                accuracy = accuracy_score(y, y_pred)
                print(f'Epoch {epoch}: Completed! Accuracy = {accuracy}')

    def predict(self, X):
        X_list = []
        for row in X.iterrows():
            doc_words = row[1]['words']
            mi_lista = []
            for word in doc_words:
                mi_lista.append(word)
            X_list.append(mi_lista)

        X_vec = self.count_vect.transform([' '.join(words) for words in X_list])
        X_vec_dense = X_vec.toarray()
        
        z = np.dot(X_vec_dense, self.W) + self.b
        y_pred = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        predicted_classes = np.argmax(y_pred, axis=1)
        return self.label_encoder.inverse_transform(predicted_classes)

# Model 3: Neural Network
class MyDataset(Dataset):
    def __init__(self, data, bow_cols):
        self.data = data
        self.bow_cols = bow_cols

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = int(self.data.loc[index, "int_class_"])
        x_bow = torch.tensor(self.data.loc[index, self.bow_cols].values.astype(float)).to(torch.float32)
        return x_bow, label

class MyNeuralNetwork(nn.Module):
    def __init__(self, dim_vocab, num_classes, dim_hidden_input, dim_hidden_output):
        super(MyNeuralNetwork, self).__init__()
        torch.manual_seed(42)
        self.first_layer = nn.Linear(dim_vocab, dim_hidden_input)
        self.hidden_layer = nn.Linear(dim_hidden_input, dim_hidden_output)
        self.last_layer = nn.Linear(dim_hidden_output, num_classes)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, xs_bow):
        first_state = self.relu(self.first_layer(xs_bow))
        hidden_state = self.relu(self.hidden_layer(first_state))
        return self.last_layer(hidden_state)

def get_loss(net, iterator, criterion):
    net.eval()
    total_loss = 0
    num_evals = 0
    with torch.no_grad():
        for xs_bow, labels in iterator:
            xs_bow, labels = xs_bow.cuda(), labels.cuda()
            logits = net(xs_bow)
            loss = criterion(logits, labels)
            total_loss += loss.item() * xs_bow.shape[0]
            num_evals += xs_bow.shape[0]
    return total_loss / num_evals

def get_preds_tests_nn(net, iterator):
    net.eval()
    preds, tests = [], []
    with torch.no_grad():
        for xs_bow, labels in iterator:
            xs_bow, labels = xs_bow.cuda(), labels.cuda()
            logits = net(xs_bow)
            soft_probs = nn.Sigmoid()(logits)
            preds += np.argmax(soft_probs.tolist(), axis=1).tolist()
            tests += labels.tolist()
    return np.array(preds), np.array(tests)