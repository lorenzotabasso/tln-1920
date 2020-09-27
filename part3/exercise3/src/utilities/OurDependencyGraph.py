import re


class OurDependencyGraph:
    """
    Class representation of the dependency graph of a sentence.
    The syntactic parser supplies a list of edges labeled with the dependency
    (example: 2 -> 6[label='nmod']) and a list of nodes labeled with the
    corresponding word in the sentence (example: 2[label='2(some)'] or
    6[label='6(toys)']).

    Attributes:
        graph: adjacent list graph, edges labeled with dependencies
        dict: dictionary {node : word of sentence}

    Methods:
        init_from_dot(dot_notation)
            Initialize graph from parser's tree supplied in dot notation.

        get_adj_neighbor(value)
            It returns a set of adjacent neighbor of node graph[key].

        get_verb_key(value)
            It returns the key of graph.dict in which the verb is stored.

    """

    def __init__(self, graph={}, dict={}):
        self.graph = graph
        self.dict = dict

    def init_from_dot(self, dot_notation):
        """
        Initialize graph from parser's tree supplied in dot notation.
        :param dot_notation: dot representation of the dependency graph
        :return: the self instance of the Dependency Graph initialized with
        values from dot_notation
        """
        lines = dot_notation.split('\n')
        lines = lines[4:-1]  # TODO: provare a commentarlo
        for line in lines:
            splits = re.split('\[label="', line, maxsplit=2)
            splits[0] = splits[0].strip()
            splits[1] = str(splits[1][:-2]).strip()
            if re.match('^[0-9]+$', splits[0]):
                labels = re.findall('\((.+)\)', splits[1])
                self.dict[int(splits[0])] = labels[0]
                self.graph[int(splits[0])] = []
            else:
                label = splits[1].strip()
                edge = splits[0].split('->')
                self.graph[int(edge[0])].append((int(edge[1]), label))

    def get_adj_neighbor(self, key):
        """
        It returns a set of adjacent neighbor of node graph[key].
        :param key: the index of the starting node inside the graph
        :return: Set of adjacent neighbor of node graph[key]
        """
        return set(self.graph[key])  # directional adj

    def get_verb_key(self, verb):
        """
        It returns the key of graph.dict in which the verb is stored.
        :param verb: the verb in input
        :return: a list of all the position of the verb in graph.dict
        """
        keys = []
        for k, v in self.dict.items():
            if v == verb:
                keys.append(k)
        return keys
