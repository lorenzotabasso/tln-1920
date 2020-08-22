import math
import sys
from datetime import datetime
from optparse import OptionParser

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from scipy import signal

# golden values for our text.
separators = [19, 50, 85]


def compute_overlap(signature, context):
    """
    Computes the number of words in common between signature and context.

    :param signature: bag of words of the text's signature (e.g. definitions +
    examples)
    :param context: bag of words of the context (e.g. sentence)
    :return: intersection between signature and context
    """

    return signature & context


def rank(x, v):
    """
    Computes rank between the vector X and the Nasari vector V

    :param x: input vector
    :param v: Nasari vector
    :return: Rank of the input vector (position)
    """

    for i in range(len(v)):
        if v[i] == x:
            return i + 1


def weighted_overlap(topic_nasari_vector, paragraph_nasari_vector):
    """
    Implementation of the Weighted Overlap metrics (Pilehvar et al.)
    :param topic_nasari_vector: Nasari vector representing the topic
    :param paragraph_nasari_vector: Nasari vector representing the paragraph
    :return: square-rooted Weighted Overlap if exist, 0 otherwise.
    """

    overlap_keys = compute_overlap(topic_nasari_vector.keys(),
                                   paragraph_nasari_vector.keys())

    overlaps = list(overlap_keys)

    if len(overlaps) > 0:
        # sum 1/(rank() + rank())
        den = sum(1 / (rank(q, list(topic_nasari_vector)) +
                       rank(q, list(paragraph_nasari_vector))) for q in overlaps)

        # sum 1/(2*i)
        num = sum(list(map(lambda x: 1 / (2 * x),
                           list(range(1, len(overlaps) + 1)))))

        return den / num

    return 0


def parse_nasari_dictionary():
    """
    It parse the Nasari input file, and it converts into a more convenient
    Python dictionary.
    :return: a dictionary representing the Nasari input file. Fomat: {word: {term:score}}
    """

    global options

    nasari_dict = {}
    with open(options.nasari, 'r', encoding="utf8") as file:
        for line in file:
            splits = line.split("\t")
            vector_dict = {}

            for term in splits[2:options.limit]:
                k = term.split("_")
                if len(k) > 1:
                    vector_dict[k[0]] = k[1]

            nasari_dict[splits[1].lower()] = vector_dict

    return nasari_dict


def read(path):
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
    for i in range(options.w, len(tokens), options.w):
        sequences.append(tokens[j:i])
        j = i

    print("\tFound {} sequences".format(str(len(sequences))))
    return sequences


def bag_of_word(tokens):
    """
    It returns the Bag of Word representation fo the given text.
    It applies lemmatization, removes the punctuation, the stop-words and duplicates.
    :param tokens: input text
    :return: Bag of Words representation of the text.
    """

    stop_words = set(stopwords.words('english'))
    punct = {',', ';', '(', ')', '{', '}', ':', '?', '!', '*'}
    wnl = nltk.WordNetLemmatizer()
    tokens = list(filter(lambda x: x not in stop_words and x not in punct, tokens))
    return set(wnl.lemmatize(t) for t in tokens)


def create_vectors(tokens, nasari):
    """
    It creates a list of Lexical Nasari vectors (a list of {term:score}).
    Every vector is linked to one token of the text.

    :param tokens: input text
    :param nasari: Nasari dictionary
    :return: list of Nasari's vectors.
    """

    tokens = bag_of_word(tokens)
    vectors = []
    for word in tokens:
        if word in nasari.keys():
            vectors.append(nasari[word])

    return vectors


def main():
    global options

    # input
    nasari = parse_nasari_dictionary()
    text = read(options.input)

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
    span = length / (options.k + 1)

    # moving average
    data = pd.DataFrame(data=y)
    short_rolling = data.rolling(window=short).mean()
    long_rolling = data.rolling(window=long).mean()
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
    now = datetime.now().strftime("Plot - %d.%m.%Y %H:%M:%S")
    plt.savefig('output/{}.png'.format(now))
    plt.show()
    print("Plot saved in output folder.")


if __name__ == "__main__":

    print('Starting segmentation...')

    argv = sys.argv[1:]
    parser = OptionParser()

    parser.add_option("-i", "--input", help='input', action="store", type="string", dest="input",
                      default="input/text.txt")

    parser.add_option("-o", "--output", help='input', action="store", type="string", dest="output",
                      default="output/")

    parser.add_option("-n", "--nasari", help='nasari file', action="store", type="string", dest="nasari",
                      default="resources/NASARI_lexical_english.txt")

    parser.add_option("-l", "--limit", help='nasari dimensions', action="store", type="int", dest="limit",
                      default="14")

    parser.add_option("-w", help='tokens sequence size', action="store", type="int", dest="w",
                      default="25")

    parser.add_option("-k", help='number of segments', action="store", type="int", dest="k",
                      default="9")

    (options, args) = parser.parse_args()

    if options.input is None or options.nasari is None or options.k is None or options.w is None:
        print("Missing mandatory parameters")
        sys.exit(2)

    main()
