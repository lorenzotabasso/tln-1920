import csv
import nltk
from nltk.corpus import stopwords
from part3.exercise2.WordNetAPIClient import WordNetAPIClient
from collections import OrderedDict



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


if __name__ == "__main__":

    options = {
        "output": "/Users/lorenzotabasso/Desktop/University/TLN/Progetto/19-20/tln-1920/part3/exercise2/input/",
    }

    content = load_data()  # Loading the content-to-form.csv file

    preprocess_content = {}
    i = 0
    for row in content:
        temp = []
        for sentence in content[row]:
            temp.append(preprocess(sentence))
        preprocess_content[i] = temp
        i += 1

    print(preprocess_content)

    wnac = WordNetAPIClient()

    last_synset = OrderedDict()

    for definitions in preprocess_content:
        for d in preprocess_content[0]:  # TODO: cercare nella stessa riga
            #print(d)
            for word in d:
                #print(word)
                synsets = wnac.get_synsets(word)
                # print(synsets)
                for s in synsets:
                    sett = preprocess(s.definition())
                    if not len(sett & d) == 0:
                        #print("\tIntersection: {}".format(sett & d))
                        last_synset[s.name()] = sett & d

    print(last_synset)
    for syn, inter in last_synset.items():
        print("{}, {}, {}".format(syn, len(inter), inter))

        print("---------------------------------------------------------------")
