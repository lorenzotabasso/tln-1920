import csv
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd


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

        return def_abstract_generic, def_concrete_generic, def_abstract_specific, def_concrete_specific


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
    :return: a list of length pow(len(definition)) containing the similarity
    score of each definition.
    """

    # list of similarity score for each type of definition (length: pow(len(definition)))
    results = []

    for d1 in definitions:
        a = preprocess(d1)  # set of terms of the first definition
        for d2 in definitions:
            b = preprocess(d2)  # set of terms of the second definition

            # Computing similarity between definitions
            t = len(a & b) / min(len(a), len(b))

            if not t == 1.0:  # A and B are the same
                results.append(t)

    # DEBUG
    # mean = np.mean(results)
    # print("MEAN: {} - RESULTS: {}".format(mean, results))

    return results


def compute_overlap_max(definitions):
    """
    It computes the overlap between the two set of the preprocessed definitions
    :param definitions: a list of definitions (strings)
    :return: a list of length |definitions| containing the maximum similarity
    score of each definition.
    """
    max_value = 0
    # list of the best similarity score for each type of definition (length: len(definitions))
    results = []

    for d1 in definitions:
        a = preprocess(d1)  # set of terms of the first definition
        for d2 in definitions:
            b = preprocess(d2)  # set of terms of the second definition

            # Computing similarity between definitions
            t = len(a & b) / min(len(a), len(b))

            if not t == 1.0 and t > max_value:
                max_value = t
        results.append(max_value)
        max_value = 0

    # DEBUG
    # mean = np.mean(results)
    # print("MEAN: {} - RESULTS: {}".format(mean, results))

    return results


if __name__ == "__main__":

    options = {
        "output": "/Users/lorenzotabasso/Desktop/University/TLN/Progetto/19-20/tln-1920/part3/exercise1/input/",
    }

    defs = load_data()  # Loading the definitions.csv file

    count = 0
    first_row = []  # generic abstract, concrete
    second_row = []  # specific abstract, concrete
    third_row = []  # generic 2 abstract, concrete
    fourth_row = []  # specific 2 abstract, concrete

    for d in defs:
        # computing the mean of the overlap of the definitions
        overlap = compute_overlap(d)
        mean = np.mean(overlap)

        overlap_max = compute_overlap_max(d)
        mean_max = np.mean(overlap_max)

        # making the percentage of the mean
        # percentage = mean * 100 / len(d)

        # filling the rows
        if count == 0:
            first_row.append('{:.0%}'.format(mean))
            third_row.append('{:.0%}'.format(mean_max))
        elif count == 1:
            first_row.append('{:.0%}'.format(mean))
            third_row.append('{:.0%}'.format(mean_max))
        elif count == 2:
            second_row.append('{:.0%}'.format(mean))
            fourth_row.append('{:.0%}'.format(mean_max))
        else:
            second_row.append('{:.0%}'.format(mean))
            fourth_row.append('{:.0%}'.format(mean_max))

        count += 1

    # build and print dataframe
    df = pd.DataFrame([first_row, second_row], columns=["Abstract", "Concrete"],
                      index=["Generic", "Specific"])
    df_max = pd.DataFrame([third_row, fourth_row], columns=["Abstract", "Concrete"],
                      index=["Generic", "Specific"])
    print(df)
    print("\nMax experiment:\n")
    print(df_max)

    # TODO: Make report.
    # Sono state aggiunte le seguenti frasi:
    # 1. The ability of one to make free choices and act by following his (NON mia)
    # 2. The ability of make independent choices
    # 3. When you feel sad for someone of you forgive him for something.

    # vedere minuto 13 - 16
    # al massimo chiedere a fede o a Telli
