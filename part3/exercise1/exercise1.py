import csv
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    """
    It reads che definition's CSV
    :return: four list containing the read definitions.
    """
    with open(options["output"], "r", encoding="utf-8") as definitions:
        reader = csv.reader(definitions, delimiter=';')

        def_abstract_generic = []
        def_concrete_generic = []
        def_abstract_specific = []
        def_concrete_specific = []

        first = True
        for line in reader:
            if not first:
                def_abstract_generic.append(line[2])
                def_concrete_generic.append(line[0])
                def_abstract_specific.append(line[3])
                def_concrete_specific.append(line[1])
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
    tokens = list(
        filter(lambda x: x not in stop_words and x not in punct, tokens))

    # Lemmatization
    lemmatized_tokens = set(wnl.lemmatize(t) for t in tokens)

    return lemmatized_tokens


def compute_overlap_terms(definitions):
    """
    It computes the overlap between the two set of the preprocessed terms
    :param definitions: a list of definitions (strings)
    :return: a list of length pow(len(definition)) containing the similarity
    score of each definition.
    """

    # list of similarity score for each type of definition (length: pow(len(definition)))
    results = []

    i = 0
    while i < len(definitions):
        a = preprocess(definitions[i])  # set of terms of the first definition
        j = i + 1
        while j < len(definitions) - 1:
            # print(i,j)  # DEBUG
            b = preprocess(definitions[j])  # set of terms of the second definition
            # Computing similarity between definitions
            t = len(a & b) / min(len(a), len(b))
            results.append(t)
            j = j + 1

        i = i + 1

    return results


def compute_overlap_pos(definitions):
    """
    It computes the overlap between the two set of the preprocessed definitions 
    converted in POS tagging.
    :param definitions: a list of definitions (strings)
    :return: a list of length |definitions| containing the maximum similarity
    score of each definition.
    """

    results = []

    i = 0
    while i < len(definitions):
        text1 = word_tokenize(definitions[i])
        temp_a = nltk.pos_tag(text1)
        a = set(x[1] for x in temp_a)

        j = i + 1
        while j < len(definitions) - 1:
            # print(i,j)  # DEBUG
            text2 = word_tokenize(definitions[j])
            temp_b = nltk.pos_tag(text2)
            b = set(y[1] for y in temp_b)

            # Computing similarity between definitions
            # intersec = [z for z in a if z in b]
            t = len(a & b) / min(len(a), len(b))  # normalization step
            results.append(t)
            j = j + 1

        i = i + 1

    return results


if __name__ == "__main__":

    options = {
        "output": "./input/definitions.csv"
    }

    defs = load_data()  # Loading the definitions.csv file

    count = 0
    first_row = []  # generic abstract, concrete
    second_row = []  # specific abstract, concrete
    third_row = []  # generic 2 abstract, concrete
    fourth_row = []  # specific 2 abstract, concrete

    percentage1 = {
        "generic_abstract": 0,
        "generic_concrete": 0,
        "specific_abstract": 0,
        "specific_concrete": 0
    }

    percentage2 = {
        "generic_abstract": 0,
        "generic_concrete": 0,
        "specific_abstract": 0,
        "specific_concrete": 0
    }

    for d in defs:
        # computing the mean of the overlap of the definitions
        overlap_terms = compute_overlap_terms(d)
        mean_terms = np.mean(overlap_terms)

        overlap_pos = compute_overlap_pos(d)
        mean_pos = np.mean(overlap_pos)

        # making the percentage of the mean
        # percentage = mean * 100 / len(d)

        # filling the rows
        if count == 0:
            first_row.append('{:.0%}'.format(mean_terms))
            percentage1["generic_abstract"] = mean_terms
            third_row.append('{:.0%}'.format(mean_pos))
            percentage2["generic_abstract"] = mean_pos
        elif count == 1:
            first_row.append('{:.0%}'.format(mean_terms))
            percentage1["generic_concrete"] = mean_terms
            third_row.append('{:.0%}'.format(mean_pos))
            percentage2["generic_concrete"] = mean_pos
        elif count == 2:
            second_row.append('{:.0%}'.format(mean_terms))
            percentage1["specific_abstract"] = mean_terms
            fourth_row.append('{:.0%}'.format(mean_pos))
            percentage2["specific_abstract"] = mean_pos
        else:
            second_row.append('{:.0%}'.format(mean_terms))
            percentage1["specific_concrete"] = mean_terms
            fourth_row.append('{:.0%}'.format(mean_pos))
            percentage2["specific_concrete"] = mean_pos

        count += 1

    # build and print dataframe
    df = pd.DataFrame([first_row, second_row], columns=["Abstract", "Concrete"],
                      index=["Generic", "Specific"])
    df_max = pd.DataFrame([third_row, fourth_row], columns=["Abstract", "Concrete"],
                          index=["Generic", "Specific"])
    print(df)
    print("\nPOS Experiment:\n")
    print(df_max)

    # Pandas Print -------------------------------------------------------------

    # Baseline
    print1 = [[percentage1["generic_abstract"], percentage1["generic_concrete"]],
              [percentage1["specific_abstract"], percentage1["specific_concrete"]]]
    df1 = pd.DataFrame(print1, columns=["Abstract", "Concrete"],
                      index=["Generic", "Specific"])
    df1.plot.bar()
    plt.xticks(rotation=30, horizontalalignment="center")
    plt.title("Baseline")
    plt.show()

    # POS Experiment
    print2 = [[percentage2["generic_abstract"], percentage2["generic_concrete"]],
              [percentage2["specific_abstract"], percentage2["specific_concrete"]]]
    df2 = pd.DataFrame(print2, columns=["Abstract", "Concrete"],
                       index=["Generic", "Specific"])
    df2.plot.bar()
    plt.xticks(rotation=30, horizontalalignment="center")
    plt.title("POS Experiment")
    plt.show()

    # TODO: Make report.
    # Sono state aggiunte le seguenti frasi:
    # 1. The ability of one to make free choices and act by following his (NON mia)
    # 2. The ability of make independent choices
    # 3. When you feel sad for someone of you forgive him for something.

    # vedere minuto 13 - 16
