import csv
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

from part3.exercise2.utilities.lesk import lesk


def load_data():
    """
    It reads che definition's CSV
    :return: four list containing the read definitions.
    """
    with open(options["output"] + 'content-to-form.csv', "r", encoding="utf-8") as content:
        cnt = csv.reader(content, delimiter=';')

        dictionary = {}
        i = 0
        for line in cnt:
            dictionary[i] = line
            i += 1

        return dictionary


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


if __name__ == "__main__":

    options = {
        "output": "/Users/lorenzotabasso/Desktop/University/TLN/Progetto/19-20/tln-1920/part3/exercise2/input/",
    }

    content = load_data()  # Loading the content-to-form.csv file

    final_test = {}
    final = {}

    for index in content:
        max = ("", 0)
        for definition in content[index]:
            for word in preprocess(definition):
                common = set()

                for s in wn.synsets(word):
                    common.add(s.name())

                for synset_name in common:
                    best_synset = lesk(preprocess_synset(synset_name), definition)
                    value = best_synset.name()

                    if not value in final_test:
                        final_test[value] = 1
                    else:
                        final_test[value] += 1
                        if final_test[value] > max[1]:
                            max = (value, final_test[value])

        final[index] = max
        print("INDEX: {}, MAX: {}".format(index, max))
        final_test = {}

    print(final)
