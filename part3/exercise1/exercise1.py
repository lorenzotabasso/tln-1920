import csv
from datetime import datetime

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    """
    It reads che definition's CSV
    :return: four list containing the read definitions.
    """
    with open(config["output"], "r", encoding="utf-8") as definitions:
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


def compute_overlap_cosine(definitions):
    """
    It computes the overlap between the two set of the preprocessed definitions 
    using cosine similarity.
    :param definitions: a list of definitions (strings)
    :return: a list of length |definitions| containing the maximum similarity
    score of each definition.
    """
    
    # Preprocess step.
    # It returns original array preprocessed without eliminating duplicates (no Sets)
    clean_defs = []
    for d in definitions:
        # Removing stopwords
        d = d.lower()
        stop_words = set(stopwords.words('english'))
        punct = {',', ';', '(', ')', '{', '}', ':', '?', '!', '.'}
        wnl = nltk.WordNetLemmatizer()
        tokens = nltk.word_tokenize(d)
        tokens = list(
            filter(lambda x: x not in stop_words and x not in punct, tokens))

        # Lemmatization
        lemmatized_tokens = ' '.join(list(wnl.lemmatize(t) for t in tokens))

        clean_defs.append(lemmatized_tokens)
    
    '''
    CountVectorizer will create k vectors in n-dimensional space, where:
    - k is the number of sentences,
    - n is the number of unique words in all sentences combined.
    If a sentence contains a certain word, the value will be 1 and 0 otherwise
    '''
    vectorizer = CountVectorizer().fit_transform(clean_defs)
    vectors = vectorizer.toarray()

    results = []
    i = 0
    while i < len(vectors):
        a = vectors[i]
        j = i + 1
        while j < len(definitions) - 1:
            # print(i,j)  # DEBUG
            b = vectors[j]

            # Computing cosine similarity between definitions.
            # Cosine_similarity() expect 2D arrays, and the input vectors are 
            # 1D arrays, so we need reshaping.
            a = a.reshape(1, -1)
            b = b.reshape(1, -1)
            res = cosine_similarity(a, b)[0][0]
            
            results.append(res)
            j = j + 1

        i = i + 1

    return results
    

if __name__ == "__main__":

    config = {
        "output": "./input/definitions.csv"
    }

    defs = load_data()  # Loading the definitions.csv file

    count = 0
    first_row = []  # generic abstract, concrete
    second_row = []  # specific abstract, concrete
    third_row = []  # generic 2 abstract, concrete
    fourth_row = []  # specific 2 abstract, concrete
    fifth_row = []  # generic 3 abstract, concrete
    sixth_row = []  # specific 3 abstract, concrete

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

    percentage3 = {
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

        overlap_cosine = compute_overlap_cosine(d)
        mean_cosine = np.mean(overlap_cosine)

        # filling the rows
        if count == 0:
            first_row.append('{:.0%}'.format(mean_terms))
            percentage1["generic_abstract"] = mean_terms
            third_row.append('{:.0%}'.format(mean_pos))
            percentage2["generic_abstract"] = mean_pos
            fifth_row.append('{:.0%}'.format(mean_cosine))
            percentage3["generic_abstract"] = mean_cosine
        elif count == 1:
            first_row.append('{:.0%}'.format(mean_terms))
            percentage1["generic_concrete"] = mean_terms
            third_row.append('{:.0%}'.format(mean_pos))
            percentage2["generic_concrete"] = mean_pos
            fifth_row.append('{:.0%}'.format(mean_cosine))
            percentage3["generic_concrete"] = mean_cosine
        elif count == 2:
            second_row.append('{:.0%}'.format(mean_terms))
            percentage1["specific_abstract"] = mean_terms
            fourth_row.append('{:.0%}'.format(mean_pos))
            percentage2["specific_abstract"] = mean_pos
            sixth_row.append('{:.0%}'.format(mean_cosine))
            percentage3["specific_abstract"] = mean_cosine
        else:
            second_row.append('{:.0%}'.format(mean_terms))
            percentage1["specific_concrete"] = mean_terms
            fourth_row.append('{:.0%}'.format(mean_pos))
            percentage2["specific_concrete"] = mean_pos
            sixth_row.append('{:.0%}'.format(mean_cosine))
            percentage3["specific_concrete"] = mean_cosine

        count += 1

    # build and print dataframe
    df_baseline = pd.DataFrame([first_row, second_row], columns=["Abstract", "Concrete"],
                               index=["Generic", "Specific"])
    df_pos = pd.DataFrame([third_row, fourth_row], columns=["Abstract", "Concrete"],
                          index=["Generic", "Specific"])
    df_cosine = pd.DataFrame([fifth_row, sixth_row], columns=["Abstract", "Concrete"],
                             index=["Generic", "Specific"])

    print("\n\nBaseline:\n")
    print(df_baseline)
    print("\nPOS Experiment:\n")
    print(df_pos)
    print("\nCosine Similarity Experiment:\n")
    print(df_cosine)

    # Pandas Print -------------------------------------------------------------

    # Baseline
    print1 = [[percentage1["generic_abstract"], percentage1["generic_concrete"]],
              [percentage1["specific_abstract"], percentage1["specific_concrete"]]]
    df1 = pd.DataFrame(print1, columns=["Abstract", "Concrete"],
                      index=["Generic", "Specific"])
                      
    df1.plot.bar()
    plt.xticks(rotation=30, horizontalalignment="center")
    plt.title("Baseline")
    plt.xlabel("Concepts")
    plt.ylabel("Similarity (higher is better)")

    # saving plot in output folder
    now = datetime.now().strftime("Baseline - %d.%m.%Y-%H:%M:%S")  # dd/mm/YY-H:M:S
    plt.savefig('output/{}.png'.format(now))
    plt.show()
    print("Baseline's plot saved in output folder.")

    # POS Experiment
    print2 = [[percentage2["generic_abstract"], percentage2["generic_concrete"]],
              [percentage2["specific_abstract"], percentage2["specific_concrete"]]]
    df2 = pd.DataFrame(print2, columns=["Abstract", "Concrete"],
                       index=["Generic", "Specific"])
    df2.plot.bar()
    plt.xticks(rotation=30, horizontalalignment="center")
    plt.title("POS Experiment")
    plt.xlabel("Concepts")
    plt.ylabel("Similarity (higher is better)")

    # saving plot in output folder
    now = datetime.now().strftime("POS - %d.%m.%Y-%H:%M:%S")  # dd/mm/YY-H:M:S
    plt.savefig('output/{}.png'.format(now))
    plt.show()
    print("POS's plot saved in output folder.")

    # Cosine Similarity Experiment
    print3 = [[percentage3["generic_abstract"], percentage3["generic_concrete"]],
              [percentage3["specific_abstract"], percentage3["specific_concrete"]]]
    df3 = pd.DataFrame(print3, columns=["Abstract", "Concrete"],
                       index=["Generic", "Specific"])
    df3.plot.bar()
    plt.xticks(rotation=30, horizontalalignment="center")
    plt.title("Cosine Similarity Experiment")
    plt.xlabel("Concepts")
    plt.ylabel("Similarity (higher is better)")

    # saving plot in output folder
    now = datetime.now().strftime("Cosine Similarity - %d.%m.%Y-%H:%M:%S")  # dd/mm/YY-H:M:S
    plt.savefig('output/{}.png'.format(now))
    plt.show()
    print("Cosine Similarity's plot saved in output folder.")

    # TODO: fare le cloud word?

    # TODO: Make report.
    # Sono state aggiunte le seguenti frasi:
    # 1. The ability of one to make free choices and act by following his (NON mia)
    # 2. The ability of make independent choices
    # 3. When you feel sad for someone of you forgive him for something.
