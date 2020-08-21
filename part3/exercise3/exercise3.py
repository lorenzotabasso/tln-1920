"""
Given a transitive verb, we search N sentences in the Brown corpus which 
contains the given verb. Then, we do WSD (using Lesk algorithm) on the verb 
arguments (subj and obj), and finally, we compute the Filler's supersense
incidence rate.
"""

import sys
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer

from part3.exercise3.utilities.DependencyGraph import DependencyGraph
from part3.exercise3.utilities.lesk import *

# global parameter for WordNet search
verbs_pos = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
subj_dept = ['nsubj', 'nsubjpass']
obj_dept = ['dobj', 'iobj']


def text_extraction(verb):
    """
    It analyzes the Brown Corpus and extracts the sentences containing the
    desired verb given as global input.
    :return: list of sentences (in which each sentence is a list of words)
    """

    lemmatizer = WordNetLemmatizer()
    list_sent = brown.sents()

    # if you want to filter sentences by category
    # list_sent = brown.sents(categories=['news'])

    sentences = []
    for sent in list_sent:
        tags = dict(nltk.pos_tag(sent))
        for word in sent:
            if tags[word] in verbs_pos:
                word = lemmatizer.lemmatize(word, 'v')
                if word == verb:
                    sentences.append(sent)

    return sentences


def lemmatize(graph, tags):
    """
    Apply lemmatization to verbs in the dependency graph
    :param graph: dependency graph
    :param tags: PoS of corresponding sentence
    :return: new and lemmatized dependency graph
    """

    lemmatizer = WordNetLemmatizer()
    new_dict = {}
    for k in graph.dict:
        word = graph.dict[k]
        if word in tags.keys() and tags[word] in verbs_pos:
            new_dict[k] = lemmatizer.lemmatize(word, 'v')
        else:
            new_dict[k] = word

    return DependencyGraph(graph.graph, new_dict)


def hanks(verb):
    """
    Implementation of P. Hanks theory
    """

    fillers = []  # [(subj, obj, sentence)]
    sentences = []

    # Set the URI to communicate with Stanford CoreNLP
    dependency_parser = CoreNLPDependencyParser(url="http://localhost:9000")

    print('[1] - Extracting sentences...')
    list_word_sentences = text_extraction(verb)
    for sent in list_word_sentences:
        sentence = ' '.join(sent)
        sentences.append(sentence.strip())

    sentences = [x.lower() for x in sentences]
    print("\t{} sentences in which the verb \'{}\' appears.".format(str(len(sentences)), verb))

    print('\n[2] - Extracting fillers...')
    for sentence in sentences:
        # PoS Tagging
        sentence = sentence.replace('.', '')
        tokens = nltk.word_tokenize(sentence)
        tags = dict(nltk.pos_tag(tokens))

        # Syntactic parsing
        result = dependency_parser.raw_parse(sentence)
        dep = next(result)
        graph = DependencyGraph()
        graph.from_dot(dep.to_dot())

        # Lemmatization
        lemmatized_graph = lemmatize(graph, tags)
        index = lemmatized_graph.get_index(verb)
        if len(index) <= 0:
            print("\tError in **{}**".format(sentence), file=sys.stderr)
            continue

        # Adjacency List
        adjs = lemmatized_graph.get_directional_adj(index[0])
        adjs = list(filter(lambda x: x[1] in subj_dept or x[1] in obj_dept, adjs))

        # Valency = 2
        if len(adjs) == 2:
            if adjs[0][1] in subj_dept:
                w1 = lemmatized_graph.dict[adjs[0][0]]
                w2 = lemmatized_graph.dict[adjs[1][0]]
            else:
                w1 = lemmatized_graph.dict[adjs[1][0]]
                w2 = lemmatized_graph.dict[adjs[0][0]]
            fillers.append((w1, w2, sentence))  # where w1 = subj and w2 = obj

    tot = len(fillers)
    print("\n[3] - Total of {} Fillers".format(str(tot)))
    for f in fillers:
        print("\t{}".format(f))

    semantic_types = {}  # {(s1, s2): count}
    for f in fillers:
        # WSD
        s1 = lesk(f[0], f[2])
        s2 = lesk(f[1], f[2])
        if s1 is not None and s2 is not None:
            # Supersences
            t = (s1.lexname(), s2.lexname())

            # Frequency
            if t in semantic_types.keys():
                semantic_types[t] = semantic_types[t] + 1
            else:
                semantic_types[t] = 1

    print('[4] - Finding Semantic Clusters (percentage, count of instances, semantic cluster):')
    for key, value in sorted(semantic_types.items(), key=lambda x: x[1]):
        to_print = str(round((value / tot) * 100, 2))
        print("\t[{}%] - {} - {}".format(to_print, value, key))


if __name__ == "__main__":
    """
    IMPORTANT: Before run, make sure to download Stanford CoreNLP tool and run 
    it using the following command inside its root folder. 
    
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
    
    After that, you can run this exercise.
    """

    # take, put, give, get, meet
    verb = input("Enter a verb to search in the Brown corpus: ")
    hanks(verb)
