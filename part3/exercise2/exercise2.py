import csv
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import CountVectorizer

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


if __name__ == "__main__":

    options = {
        "output": "/Users/lorenzotabasso/Desktop/University/TLN/Progetto/19-20/tln-1920/part3/exercise2/input/",
    }

    content = load_data()  # Loading the content-to-form.csv file

    '''
    1. prendo definzione, disambiguo con pos-tagging. il primo nome è il genus
    2. come approccio personalizzato, teveno un dizionario di genus (dopo aver esplorato tutte le definizioni) e espandevo solo il genus più frequente
    riducendo la ricerca
    3. prendo da wordnet i synsets di quel sostantivo, e per ognuno di essi parto in basso con gli iponimi
    4. calcolo l'iponimo dell'iponimo dell'iponimo..., per non sclerare utilizza la closure (chiusura trasitiva). Calcolo gli iponimi fino a un certo 
    livello
    5. calcola iponimo con più overlapping, stili classifica
    '''

    for index in content:
        # for index in range(1):

        hyponyms_list = []

        for definition in content[index]:
            # for definition in content[0]:
            local_genus = {}
            hyponyms = []

            def_tokens = word_tokenize(definition)
            results = nltk.pos_tag(def_tokens)

            possibles_genus = list(filter(lambda x: x[1] == "NN", results))
            # Es.: [('abstract', 'NN'), ('concept', 'NN'), ('idea', 'NN'), ('fairness', 'NN'), ('front', 'NN'), ('code', 'NN'), ('community', 'NN')]

            for g in possibles_genus:
                if not g[0] in local_genus:
                    local_genus[g[0]] = 1
                else:
                    local_genus[g[0]] += 1

            #         print(index, genus)
            #         print("{} - {}\n".format(index, local_genus))

            if len(local_genus) > 0:
                genus = max(local_genus, key=local_genus.get)
                #             print("GENUS: " + genus)

                syns = wn.synsets(genus)

                # Prendiamo tutti gli iponimi per il genus della singola definizione
                for i, s in enumerate(syns, start=0):
                    hypon = lambda s: s.hyponyms()  # SOTTONOME, significato semantico incluso in altra parola
                    all_hypon = list(s.closure(hypon, depth=10))  # TODO: aumentare a 2,3
                    hyponyms.extend([x.name().split(".")[0] for x in all_hypon])
            #                 print("SYN: {} \t HYPER: {}".format(s,t))

            #             print(index, hyponyms, "\n")
            #         else:
            #             print("NADA")

            hyponyms_list.append(' '.join(hyponyms))

        #     print(hyponyms_list)

        '''
        CountVectorizer will create k vectors in n-dimensional space, where:
        - k is the number of sentences,
        - n is the number of unique words in all sentences combined.
        If a sentence contains a certain word, the value will be 1 and 0 otherwise
        '''

        vectorizer = CountVectorizer()
        matrix = vectorizer.fit_transform(hyponyms_list)

        feature_list = vectorizer.get_feature_names()
        vectors = matrix.toarray()

        m = vectors.sum(axis=0).argmax()

        print(m)
        print(feature_list[m] + '\n')
    #     print(feature_list)
