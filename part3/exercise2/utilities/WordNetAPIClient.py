from nltk.corpus import wordnet as wn


class WordNetAPIClient:
    """
    This class implements all the possible operations (API) for accessing to
    WordNet.
    """

    def get_synsets(self, word):
        """
        :param word: word for which we need to find meaning
        :return: Synset list associated to the given word
        """
        return wn.synsets(word)
