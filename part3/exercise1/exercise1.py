import csv
import nltk
from nltk.corpus import stopwords
import numpy as np


def load_data():
    """
    It reads che definition's CSV
    :return: four list containing the read definitions.
    """
    with open(options["output"] + 'definitions.csv', "r", encoding="utf-8") as definitions:
        reader = csv.reader(definitions, delimiter=';')

        def_concrete_generic = []
        def_concrete_specific = []
        def_abstract_generic = []
        def_abstract_specific = []

        first = True
        for line in reader:
            if not first:
                def_concrete_generic.append(line[0])
                def_concrete_specific.append(line[1])
                def_abstract_generic.append(line[2])
                def_abstract_specific.append(line[3])
            else:
                first = False

        return def_concrete_generic, def_concrete_specific, def_abstract_generic, def_abstract_specific


def preprocess(definition):
    """
    It does some preprocess: removes the stopword, punctuation and does the
    lemmatization of the tokens inside the sentence.
    :param definition: a string representing a definition
    :return: a set of string which contains the preprocessed string tokens.
    """

    # Removing stopwords
    definition = definition.lower()
    stop_words = set(stopwords.words('english'))
    punct = {',', ';', '(', ')', '{', '}', ':', '?', '!', '.'}
    wnl = nltk.WordNetLemmatizer()
    tokens = nltk.word_tokenize(definition)
    tokens = list(filter(lambda x: x not in stop_words and x not in punct, tokens))

    # Lemmatization
    lemmatized_tokens = set(wnl.lemmatize(t) for t in tokens)

    return lemmatized_tokens


def compute_overlap(definitions):
    """
    It computes the overlap between the two set of the preprocessed definitions
    :param definitions: a list of definitions (strings)
    :return: a list containing the similarity score of each definition.
    """
    temp = 0
    results = []  # list of the best similarity score for each type of definition

    for d1 in definitions:
        a = preprocess(d1)  # set of terms of the first definition
        for d2 in definitions:
            b = preprocess(d2)  # set of terms of the second definition

            # Computing similarity between definitions
            k = len(b & a) / min(len(d1), len(d2))

            if k > temp:
                temp = k
        results.append(temp)

    return results


if __name__ == "__main__":

    options = {
        "output": "/Users/lorenzotabasso/Desktop/University/TLN/Progetto/19-20/tln-1920/part3/exercise1/input/",
    }

    defs = load_data()

    # The ability of one to make free choices and act by following his (NON mia)
    # The ability of make independent choices
    # When you feel sad for someone of you forgive him for something.

    count = 0

    for d in defs:
        mean = np.mean(compute_overlap(d))

        index = ""
        if count == 0:
            index = "Concrete Generic"
        elif count == 1:
            index = "Concrete Specific"
        elif count == 2:
            index = "Abstract Generic"
        else:
            index = "Abstract Specific"

        print("{}: {:.3f}".format(index, mean))
        count += 1

        # TODO: Make report.
