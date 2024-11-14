# This is a Natural Language Processing assignment that implements various text processing techniques including:
# - Tokenization
# - Stemming and Stopwords
# - Bag of Words
# - TF-IDF
# - Cosine Similarity
# - N-grams
# - Perplexity calculation
# - Linear Interpolation

import re
import pandas as pd
import numpy as np
from nltk.tokenize import wordpunct_tokenize

def get_tokens(text):
    pattern = r"[a-zA-ZáéíóúÁÉÍÓÚüÜñÑ0-9]+|[()¿?\[\],.:'\"\-¡!]+"
    tokens = re.findall(pattern, text)
    return tokens

def get_vocab(corpus):
    tokens = []
    for sentence in corpus:
        sentence_tokens = get_tokens(sentence)
        filtered_tokens = [token for token in sentence_tokens if token.isalnum()]
        tokens.extend(filtered_tokens)
        unique_tokens = list(set(tokens))
    return unique_tokens

def pre_processing(vocabulario, idioma):
    stemming_rules = {
        'espanol': ['ar', 'er', 'ir', 'ando', 'iendo', 'ado', 'ido', 'o', 'as', 'a', 'amos', 'áis', 'an', 'e', 's'],
        'ingles': ['ing', 'ed','ied', 's']
    }
    stopwords = {
        'espanol': ['y', 'el', 'la', 'lo' ,'los', 'las', 'un', 'una', 'unos', 'unas', 'o', 'de', 'en', 'a', 'para', 'que', 'qué', 'yo'],
        'ingles': ['the', 'and', 'but', 'is', 'are', 'or', 'you', 'your', 'what', 's', 'do', 'on', 'that', 've']
    }

    if idioma not in stemming_rules or idioma not in stopwords:
        raise ValueError("Idioma no soportado")
    else:
        vocab_procesado = set()
        for palabra in vocabulario:
            if palabra.lower() not in stopwords.get(idioma):
                for sufijo in stemming_rules[idioma]:
                    if palabra.endswith(sufijo):
                        palabra = palabra[: -len(sufijo)]
                if len(palabra) > 1:
                    vocab_procesado.add(palabra)
        return list(vocab_procesado)

def bag_of_words(corpus):
    vocabulary = set(word for doc in corpus for word in doc.split())
    word_freq = {}
    for word in vocabulary:
        word_freq[word] = [0] * len(corpus)
    for i, document in enumerate(corpus):
        for word in document.split():
            word_freq[word][i] += 1
    bag_of_words_df = pd.DataFrame(word_freq)
    return bag_of_words_df

def calc_tf(dataset_bow):
    max_freq = dataset_bow.max(axis=1)
    nft = dataset_bow.div(max_freq, axis=0)
    return nft

def calc_idf(dataset_bow):
    N = len(dataset_bow)
    ni = (dataset_bow > 0).sum(axis=0)
    idf = np.log10(N / ni)
    idf_dict = {word: idf_value for word, idf_value in zip(dataset_bow.columns, idf)}
    return idf_dict

def calc_tf_idf(tf, idf):
    tf_idf = tf * idf
    return tf_idf

def cosine_similarity(v1, v2):
    num = v1.dot(v2)
    dist1 = np.linalg.norm(v1)
    dist2 = np.linalg.norm(v2)
    den = dist1 * dist2
    result = num / den
    return result

def get_sentences(texto):
    lines = texto.split("\n")
    sentences = []
    for line in lines:
        pattern = r"[a-zA-ZáéíóúÁÉÍÓÚüÜñÑ]+"
        words = re.findall(pattern, line)
        if len(words) > 0:
            line = "* " + line
            sentences.append(line)
    return sentences

def n_grams(corpus, n=3):
    n_grams = {}
    for sentence in corpus:
        words = sentence.split()
        sentece_padding = ["*"] * (n - 1) + words + ["STOP"]
        for i in range(len(sentece_padding) - n +1):
            n_gram = tuple(sentece_padding[i:i+n])
            if n_gram in n_grams:
                n_grams[n_gram] += 1
            else:
                n_grams[n_gram] = 1
    return n_grams

def get_q_cond(sentence, index ,n):
    q_cond = []
    count = n - 1
    places = 1
    while count > 0:
        element_before = sentence[index - places]
        q_cond.append(element_before)
        places = places + 1
        count = count - 1
    return q_cond

def get_probability(sentence, n_grams_frequency, n):
    probability = 1.0
    words = sentence.split()
    sentence_padding = ["*"] * (n - 1) + words

    for i in range(len(sentence_padding) - n + 1):
        index = i + (n-1)
        q_prob = tuple(sentence_padding[i:i+n])
        q_cond = q_prob[:-1]

        if q_prob not in n_grams_frequency or q_cond not in n_grams_frequency:
            return 0.0

        count_numerador = n_grams_frequency[q_prob]

        if n != 1:
            count_denominador = n_grams_frequency[q_cond]
        else:
            count_denominador = len(words)
        probability *= count_numerador / count_denominador
    return probability

def n_grams_recursive(corpus, n):
    n_grams_recursive = {}
    for i in range(n,0,-1):
        n_grams_recursive.update(n_grams(corpus,i))
    return n_grams_recursive

def get_perplexity(corpus, n):
    M = len(get_vocab(corpus))
    n_grams_frequency = n_grams_recursive(corpus, n)
    sum = 0
    for sentence in corpus:
        p = np.log2(get_probability(sentence, n_grams_frequency, n))
        sum = sum + p
    l = (1/M) * sum
    return (2**(-l))

def get_probability_lineal_interpol(sentence, corpus, l_1, l_2, l_3):
    n_grams_frequency = n_grams_recursive(corpus, 3)
    sum_3 = l_3 * get_probability(sentence, n_grams_frequency, 3)
    sum_2 = l_2* get_probability(sentence, n_grams_frequency, 2)
    sum_1 = l_1 * get_probability(sentence, n_grams_frequency, 1)
    probability = sum_1 + sum_2 + sum_3
    return probability

def get_pp_interpol(corpus, l_1, l_2, l_3):
    M = len(get_vocab(corpus))
    sum = 0
    for sentence in corpus:
        p = np.log2(get_probability_lineal_interpol(sentence, corpus, l_1,l_2,l_3))
        sum = sum + p
    l = (1/M) * sum
    return (2**(-l))