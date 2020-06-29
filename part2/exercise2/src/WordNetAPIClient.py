from nltk.corpus import wordnet as wn


class WordNetAPIClient:
    """
    This class implements all the possible operations (API) for accessing to
    WordNet.
    """

    def get_disambiguation_context(self, word):
        """
        :param word: word for which we need to find meaning
        :return: a dictionary of Synset associated to the given word
        """
        synsets = wn.synsets(word)
        ret = {}  # return variable

        for s in synsets:
            if s.examples():
                t = [s.lemma_names()[0], s.examples()[0]]

                i = 0
                for hypo in s.hyponyms():
                    if i == 3:
                        break
                    if hypo.lemma_names():
                        t.append(hypo.lemma_names()[0])
                    if hypo.examples():
                        t.append(hypo.examples()[0])
                    i += 1

                i = 0
                for hyper in s.hypernyms():
                    if i == 3:
                        break
                    if hyper.lemma_names():
                        t.append(hyper.lemma_names()[0])
                    if hyper.examples():
                        t.append(hyper.examples()[0])
                    i += 1
            else:
                t = [s.lemma_names()[0], []]

                i = 0
                for hypo in s.hyponyms():
                    if i == 3:
                        break
                    if hypo.lemma_names():
                        t.append(hypo.lemma_names()[0])
                    if hypo.examples():
                        t.append(hypo.examples()[0])
                    i += 1

                i = 0
                for hyper in s.hypernyms():
                    if i == 3:
                        break
                    if hyper.lemma_names():
                        t.append(hyper.lemma_names()[0])
                    if hyper.examples():
                        t.append(hyper.examples()[0])
                    i += 1

            ret[s.name()] = t

        return ret
