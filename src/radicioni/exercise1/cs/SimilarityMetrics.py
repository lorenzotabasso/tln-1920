from math import log


class SimilarityMetrics:
    """
    This class contains the implementations of the three similarity metrics.
    """

    def __init__(self, wordnet_api_client):
        self.wnac = wordnet_api_client

    def wu_palmer_metric(self, synset1, synset2):
        """
        Implementations of the Wu-Palmer metric.
        """
        lcs = self.wnac.lowest_common_subsumer(synset1, synset2)
        if lcs is None:
            return 0

        depth_lcs = self.wnac.depth_path(lcs, lcs)
        depth_s1 = self.wnac.depth_path(synset1, lcs)
        depth_s2 = self.wnac.depth_path(synset2, lcs)
        result = (2 * depth_lcs) / (depth_s1 + depth_s2)
        return result * 10

    def shortest_path_metric(self, synset1, synset2):
        """
        Implementations of the Shortest Path metric.
        """
        max_depth = self.wnac.depth_max
        len_s1_s2 = self.wnac.distance(synset1, synset2)
        if len_s1_s2 is None:
            return 0
        res = 2 * max_depth - len_s1_s2
        return (res / 40) * 10

    def leakcock_chodorow_metric(self, synset1, synset2):
        """
        Implementations of the Leakcock-Chodorow metric.
        """
        max_depth = self.wnac.depth_max
        len_s1_s2 = self.wnac.distance(synset1, synset2)
        if len_s1_s2 is None:
            return 0
        if len_s1_s2 == 0:
            len_s1_s2 = 1
            res = -(log((len_s1_s2 / ((2 * max_depth) + 1)), 10))
        else:
            res = -(log((len_s1_s2 / (2 * max_depth)), 10))
        return (res / (log(2 * self.wnac.depth_max + 1, 10))) * 10

    def get_all(self):
        """
        It returns a list of reference to the metrics implementation inside this class.
        """
        return [(self.wu_palmer_metric, "Wu & Palmer"), (self.shortest_path_metric, "Shortest Path"),
                (self.leakcock_chodorow_metric, "Leakcock & Chodorow")]
