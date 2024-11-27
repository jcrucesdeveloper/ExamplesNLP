# Tu respuesta

from itertools import chain
from functools import partial, lru_cache
from math import log

# El vocabulario deberia estar escondido para los estudiantes

# Function to get de prior of a class
# Count the documents in a class divided by the total document count
@lru_cache()
def class_prior(documents, class_):
    return len([doc for doc in documents if doc.class_ == class_]) / len(
        documents
    )


# Function to get a word likelihood given a class
@lru_cache()
def word_likelihood(documents, word, class_):
    vocab = {word for doc in documents for word in doc.words}
    superdocument = list(
        chain.from_iterable(
            [doc.words for doc in documents if doc.class_ == class_]
        )
    )
    return (superdocument.count(word) + 1) / (len(superdocument) + len(vocab))


# Hago los calculos lazy
trained_class_prior = partial(class_prior, train_set)
trained_word_likelihood = partial(word_likelihood, train_set)


def predict(documents):
    predicted_documents = []
    for doc in documents:
        # implemented using log sum instead of multiplication
        class_probabilities = [
            log(trained_class_prior(class_))
            + sum(
                [
                    log(trained_word_likelihood(word, class_))
                    for word in doc.words
                ]
            )
            for class_ in range(NUM_CLASSES)
        ]
        predicted_documents.append(
            document(
                doc.words, class_probabilities.index(max(class_probabilities))
            )
        )
    return predicted_documents


predicted = predict(test_set)
for doc in predicted:
    print(f"La clase predicha del documento {doc.words} es: {doc.class_}")
