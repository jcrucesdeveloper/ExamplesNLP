from collections import namedtuple, defaultdict
import random
from pprint import pprint

random.seed(10001)

VOCAB_SIZE = 10
NUM_CLASSES = 2
DOC_LENGTHS = tuple(range(7, 10))

TRAIN_SIZE = 8
TEST_SIZE = 1


# Create vocab
vocab = [f"w{i:02}" for i in range(VOCAB_SIZE)]
print("Vocabulary:")
pprint(vocab, compact=True)


document = namedtuple(
    "document", ("words", "class_")  # avoid python's keyword collision
)

# For each class, define a subset of words that is most common
# using a simple heuristic
# for a 6 words vocab and 2 classes this list of lists would be
# [[0, 1, 2],
#  [3, 4, 5]]
step = VOCAB_SIZE // NUM_CLASSES
split_starts = list(range(0, VOCAB_SIZE, step))[:NUM_CLASSES]
common_words_per_class = [
    list(range(VOCAB_SIZE))[i:j]
    for i, j in zip(split_starts, split_starts[1:] + [None])
]

# Crear train set, using tuples for future convinience
train_set = tuple(
    [
        document(
            words=tuple(
                random.choices(
                    vocab,
                    # asign a weight of 2 to common words a 1 for noncommon
                    weights=[
                        2 if i in common_words_per_class[class_] else 1
                        for i in range(VOCAB_SIZE)
                    ],
                    k=random.choice(DOC_LENGTHS),
                )
            ),
            class_=class_,
        )
        for class_ in random.choices(list(range(NUM_CLASSES)), k=TRAIN_SIZE)
    ]
)
print("\nTrain documents:")
pprint(train_set)

# ensure only the used words in the vocab will appear in the test set
used_words = sorted(list({word for doc in train_set for word in doc.words}))

# here word occurrence is completely random
test_set = tuple(
    [
        document(
            words=tuple(
                random.choices(used_words, k=random.choice(DOC_LENGTHS),)
            ),
            class_=None,
        )
        for _ in range(TEST_SIZE)
    ]
)

print("\nTest documents:")
pprint(test_set)
