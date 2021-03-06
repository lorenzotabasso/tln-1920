import csv
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import CountVectorizer
from nltk.wsd import lesk


def load_data():
    """
    It reads che definition's CSV

    :return: a dictionary containing all the definitions for a specified concept
    """
    with open(config["output"] + 'content-to-form.csv', "r", encoding="utf-8") as content:
        cnt = csv.reader(content, delimiter=';')

        dictionary = {}
        i = 0
        for line in cnt:
            dictionary[i] = line
            i += 1

        return dictionary


def preprocess(definition):
    """
    It does some preprocess: removes stopwords, punctuation and does the
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


def preprocess_synset(synset):
    """
    It does some preprocess: removes the stopword, punctuation and does the
    lemmatization of the tokens inside the sentence.

    :param definition: a string representing a definition
    :return: a set of string which contains the preprocessed string tokens.
    """
    pre_synset = synset.split(".")
    clean_synset = pre_synset[0]
    return clean_synset


def genus_noun(depth):
    content = load_data()  # Loading the content-to-form.csv file

    for index in content:
        hyponyms_list = []

        for definition in content[index]:
            local_genus = {}
            hyponyms = []

            def_tokens = word_tokenize(definition)
            results = nltk.pos_tag(def_tokens)

            possibles_genus = list(filter(lambda x: x[1] == "NN", results))
            # Eg.: [('abstract', 'NN'), ('concept', 'NN'), ('idea', 'NN'), ('fairness', 'NN'),
            # ('front', 'NN'), ('code', 'NN'), ('community', 'NN')]

            for g in possibles_genus:
                if not g[0] in local_genus:
                    local_genus[g[0]] = 1
                else:
                    local_genus[g[0]] += 1

            if len(local_genus) > 0:
                genus = max(local_genus, key=local_genus.get)
                syns = wn.synsets(genus)

                # We take every hyponyms for the Genus of the definition
                for i, s in enumerate(syns, start=0):
                    hypon = lambda s: s.hyponyms()
                    all_hypon = list(s.closure(hypon, depth=depth)) 
                    hyponyms.extend([x.name().split(".")[0] for x in all_hypon])

            hyponyms_list.append(' '.join(hyponyms))

        # CountVectorizer will create k vectors in n-dimensional space, where:
        # - k is the number of sentences,
        # - n is the number of unique words in all sentences combined.
        # If a sentence contains a certain word, the value will be 1 and 0 otherwise
        vectorizer = CountVectorizer()
        matrix = vectorizer.fit_transform(hyponyms_list)

        feature_list = vectorizer.get_feature_names()
        vectors = matrix.toarray()

        # Index of the element with maximum frequency across all the definition of a concept
        m = vectors.sum(axis=0).argmax()

        print("\t{} - {} - {}".format(index + 1, feature_list[m], m))


def genus_hyper(depth):
    content = load_data()  # Loading the content-to-form.csv file

    for index in content:
        genus_dict = {}
        hyponyms_list = []

        for definition in content[index]:
            hypernyms = []
            hyponyms = []
            clean_tokens = preprocess(definition)

            for word in clean_tokens:
                syn = [lesk(definition, word)]
                if len(syn) > 0:  # needed because for some words lesks returns an empty list
                    for s in syn:
                        if s:
                            hyper = lambda s: s.hypernyms()
                            all_hyper = list(s.closure(hyper, depth=depth)) 
                            hypernyms.extend([x.name().split(".")[0] for x in all_hyper])

                    for g in hypernyms:
                        if g not in genus_dict:
                            genus_dict[g] = 1
                        else:
                            genus_dict[g] += 1

            if len(genus_dict) > 0:
                genus = max(genus_dict, key=genus_dict.get)
                syns = wn.synsets(genus)

                # We take every hyponyms for the Genus of the definition
                for i, s in enumerate(syns, start=0):
                    hypon = lambda s: s.hyponyms()
                    all_hypon = list(s.closure(hypon, depth=depth))
                    hyponyms.extend([x.name().split(".")[0] for x in all_hypon])

            hyponyms_list.append(' '.join(hyponyms))

        # CountVectorizer will create k vectors in n-dimensional space, where:
        # - k is the number of sentences,
        # - n is the number of unique words in all sentences combined.
        # If a sentence contains a certain word, the value will be 1 and 0 otherwise
        vectorizer = CountVectorizer()
        matrix = vectorizer.fit_transform(hyponyms_list)

        feature_list = vectorizer.get_feature_names()
        vectors = matrix.toarray()

        # Index of the element with maximum frequency across all the definition of a concept
        m = vectors.sum(axis=0).argmax()

        print("\t{} - {} - {}".format(index + 1, feature_list[m], m))


if __name__ == "__main__":
    config = {
        "output": "/Users/lorenzotabasso/Desktop/University/TLN/Progetto/19-20/tln-1920/part3/exercise2/input/",
        "depth": 4
    }

    print("Genus Noun (depth {}):".format(config["depth"]))
    genus_noun(config["depth"])
    print("\nGenus Hyper (depth {}):".format(config["depth"]))
    genus_hyper(config["depth"])
