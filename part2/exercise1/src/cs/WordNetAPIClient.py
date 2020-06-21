from nltk.corpus import wordnet as wn


class WordNetAPIClient:
    """
    This class implements all the possible operations (API) for accessing to
    WordNet.
    """

    def __init__(self):
        """
        Costructor. Because computing the max depth of the graph is a very
        expensive task, it is computed here once for all the class.
        """
        print("[1] - Computing WordNet graph's max depth.")
        self.depth_max = self.depth_max()
        print("[1] - WordNet graph's max depth computed.")

    def depth_path(self, synset, lcs):
        """
        :param synset: synset to calculate the path
        :param lcs: Lowest Common Subsumer - the first common sense
        :return: the minimum path which contains LCS
        """

        paths = synset.hypernym_paths()
        paths = list(filter(lambda x: lcs in x, paths))  # all path containing LCS
        return min(len(path) for path in paths)

    def lowest_common_subsumer(self, synset1, synset2):
        """
        :param synset1: first synset to take LCS from
        :param synset2: second synset to take LCS from
        :return: the first common LCS
        """

        if synset1 == synset2:
            return synset1

        commons = []
        for h in synset1.hypernym_paths():
            for k in synset2.hypernym_paths():
                zipped = list(zip(h, k))  # merges 2 list in one list of tuples
                common = None
                for i in range(len(zipped)):
                    if zipped[i][0] != zipped[i][1]:
                        break
                    common = (zipped[i][0], i)

                if common is not None and common not in commons:
                    commons.append(common)

        if len(commons) <= 0:
            return None

        commons.sort(key=lambda x: x[1], reverse=True)
        return commons[0][0]

    def distance(self, synset1, synset2):
        """
        :param synset1: first synset
        :param synset2: second synset
        :return: distance between the two synset
        """

        lcs = self.lowest_common_subsumer(synset1, synset2)
        lists_synset1 = synset1.hypernym_paths()
        lists_synset2 = synset2.hypernym_paths()

        if lcs is None:
            return None

        # path from LCS to root
        lists_lcs = lcs.hypernym_paths()
        set_lcs = set()
        for l in lists_lcs:
            for i in l:
                set_lcs.add(i)
        set_lcs.remove(lcs)  # nodes from LCS (not included) to root

        # path from synset to LCS
        lists_synset1 = list(map(lambda x: [y for y in x if y not in set_lcs], lists_synset1))
        lists_synset2 = list(map(lambda x: [y for y in x if y not in set_lcs], lists_synset2))

        # path containing LCS
        lists_synset1 = list(filter(lambda x: lcs in x, lists_synset1))
        lists_synset2 = list(filter(lambda x: lcs in x, lists_synset2))

        return min(list(map(lambda x: len(x), lists_synset1))) + min(list(map(lambda x: len(x), lists_synset2))) - 2

    @staticmethod
    def depth_max():
        """
        :return: The max depth of WordNet tree (20)
        """
        return max(max(len(path) for path in sense.hypernym_paths()) for sense in wn.all_synsets())

    @staticmethod
    def get_synsets(word):
        """
        :param word: word for which we need to find meaning
        :return: Synset list associated to the given word
        """
        return wn.synsets(word)
