from collections import namedtuple
from data import Vocabulary
from batcher import Batcher
from keras.models import Sequential
from keras.layers import GRU, Dense, Embedding, Dropout
from keras.callbacks import LambdaCallback
from beam_search import beam_search

import numpy as np

vocab = Vocabulary("./vocabulary.txt")

params = {
    'hid_dim': 128,
    'emb_dim': 64,  # 64 feature for each character
    'lr': 0.05,
    'vocab_size': vocab.size,
    'keep_prob': 0.8,  # dropout value
    'batch_size': 8,  # how may examples are in input in each step
    'beam_size': 5,  # less means the net work better, more it's worst. The NN will use more time to generate output.
    'seq_len': 50,  # Maximum sequence length
    'optimizer': 'adagrad'}

param_list = list(params.keys())

hps = namedtuple("PARAM", param_list)(**params)

batcher = Batcher('./train.bin', vocab, hps)

model = Sequential()  # Keras's default starting config
# First the infos will pass into Embedding layer, then to the Dropout layer
model.add(Embedding(vocab.size, hps.emb_dim, input_length=hps.seq_len))
model.add(Dropout(1. - hps.keep_prob))
# Then we will use GRU as NN model. Unroll=true makes the NN faster in input generation. But it use more memory.
model.add(GRU(hps.hid_dim, return_sequences=True, unroll=True))
# The last layer will be a Dense (Softmax) layer
model.add(Dense(vocab.size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=hps.optimizer)


def make_name(model, vocab, hps):
    """
    Greedy function
    """

    name = []
    x = np.ones((1, hps.seq_len)) * vocab.char2id('<s>')
    i = 0

    while i < hps.seq_len:

        probs = list(model.predict(x)[0, i])
        probs = probs / np.sum(probs)
        index = np.random.choice(range(vocab.size), p=probs)
        character = vocab.id2char(index)

        if character == '\s':
            name.append(' ')
        else:
            name.append(character)

        if i >= hps.seq_len or character == '</s>':
            break
        elif i + 1 < hps.seq_len:
            x[0, i + 1] = index
        # else:
        #     x[0, i + 1] = index
        # Originariamente non c'era l'else, ma solo l'elif, da controlla re se serve o è corretto l'else

        i += 1

    print(''.join(name))


def make_name_beam(model, vocab, hps):
    """
    Beam Search
    """

    best_seq = beam_search(model, vocab, hps)
    chars = [vocab.id2char(t) for t in best_seq.tokens[1:]]
    tokens = [t if t != '\s' else ' ' for t in chars]
    tokens = ''.join(tokens)

    print(tokens)


iteration = 0
while True:

    batch = batcher.next_batch()

    model.train_on_batch(batch.input, batch.target)

    if iteration % 500 == 0:
        print('Names generated after iteration %d:' % iteration)

        # TODO: Funziona solo col greedy, perchè il greedy è non deterministico,
        # con il beam ti stampa 3 nomi uguali perchè il primo è sempre il più
        # probabile, basta commentarlo
        make_name_beam(model, vocab, hps)

        # uncomment for greedy search
        # for i in range(3):
        #     make_name(model, vocab, hps)

        print()

    iteration += 1
