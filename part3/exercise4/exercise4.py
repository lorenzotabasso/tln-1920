import nltk
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime

from part3.exercise4.utilities import create_vectors, weighted_overlap

# separator values for our text.
separators = [19, 50, 85]


def parse_nasari_dictionary():
    """
    It parse the Nasari input file, and it converts into a more convenient
    Python dictionary.

    :return: a dictionary representing the Nasari input file. Fomat: {word: {term:score}}
    """

    global options

    nasari_dict = {}
    with open(options["nasari"], 'r', encoding="utf8") as file:
        for line in file:
            splits = line.split("\t")
            vector_dict = {}

            for term in splits[2:options["limit"]]:
                k = term.split("_")
                if len(k) > 1:
                    vector_dict[k[0]] = k[1]

            nasari_dict[splits[1].lower()] = vector_dict

    return nasari_dict


def read_file(path):
    """
    It reads the input file at the specified path.

    :param path: the path to the input file.
    :return:
    """
    with open(path) as file:
        lines = file.readlines()
    return ''.join(lines)


def tokenize(text):
    """
    It divides the text in groups. Each group is composed by w words.
    :param text: input text
    :return: list of sequences (list of all groups of word)
    """
    global options

    sequences = []
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    j = 0
    for i in range(options["token_sequence_size"], len(tokens), options["token_sequence_size"]):
        sequences.append(tokens[j:i])
        j = i

    print("\tFound {} sequences".format(str(len(sequences))))
    return sequences


def segmentation():
    global options

    # input
    nasari = parse_nasari_dictionary()
    text = read_file(options["input"])

    # tokenize
    sequences = tokenize(text)

    # compute similarity neighbors
    similarities = list(np.zeros(len(sequences)))
    for i in range(1, len(sequences) - 1):
        prev = create_vectors(sequences[i - 1], nasari)
        current = create_vectors(sequences[i], nasari)
        next = create_vectors(sequences[i + 1], nasari)

        # compute square root weighted overlap
        similarity = []
        for x in prev:
            for w in current:
                similarity.append(math.sqrt(weighted_overlap(x, w)))
        left = max(similarity) if len(similarity) > 0 else 0

        similarity = []
        for x in next:
            for w in current:
                similarity.append(math.sqrt(weighted_overlap(x, w)))
        right = max(similarity) if len(similarity) > 0 else 0

        similarities[i] = (left + right) / 2

    del nasari

    print("Plotting...")

    # plot
    length = len(similarities)
    x = np.arange(0, length, 1)
    y = np.array(similarities)

    short = int(0.016 * length)
    long = int(0.08 * length)
    very_long = int(0.16 * length)
    span = length / (options["segments_number"] + 1)

    # moving average
    data = pd.DataFrame(data=y)
    # short_rolling = data.rolling(window=short).mean()
    # long_rolling = data.rolling(window=long).mean()
    ema_very_long = data.ewm(span=very_long, adjust=False).mean()

    # local minimum
    f = np.array(ema_very_long.to_numpy()).reshape((length,))
    inv_data_y = f * (-1)
    valley = signal.find_peaks_cwt(inv_data_y, np.arange(1, span))

    fig, ax = plt.subplots()
    ax.plot(x, y, label='blocks cohesion', color='c')
    plt.plot(x[valley], f[valley], "o", label="local minimum (" + str(span) + " span)", color='r')

    for x in separators:
        ax.axvline(x, color='k', linewidth=1)

    ax.set(xlabel='tokens sequence gap', ylabel='similarity',
           title='Block similarity')
    ax.grid()
    ax.legend(loc='best')

    # dd/mm/YY H:M:S
    now = datetime.now().strftime("Plot - %d.%m.%Y-%H:%M:%S")
    plt.savefig('output/{}.png'.format(now))
    plt.show()
    print("Plot saved in output folder.")


global options  # Dictionary of the configuration. Used across all the script.

if __name__ == "__main__":
    options = {
        "input": "input/snowden.txt",
        "output": "output/",
        "nasari": "resources/NASARI_lexical_english.txt",
        "limit": 14,  # limit Nasari's dimensions
        "token_sequence_size": 25,
        "segments_number": 9

    }

    print('Starting segmentation...')
    segmentation()
