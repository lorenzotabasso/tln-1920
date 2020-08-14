from nltk.corpus import wordnet as wn


def get_synsets(word):
    """
    :param word: word for which we need to find meaning
    :return: Synset list associated to the given word
    """
    return wn.synsets(word)


if __name__ == "__main__":
    synsets = get_synsets("moral")

    for s in synsets:
        print(s.name())
        print(s.definition())
        print(s.examples())
        print("---------------------------------------------------------------")
