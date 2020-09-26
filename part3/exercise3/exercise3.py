import sys
import nltk
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer
from nltk.wsd import lesk

from part3.exercise3.utilities.OurDependencyGraph import OurDependencyGraph
from part3.exercise3.utilities.our_lesk import our_lesk

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
                # The word is a verb, so we apply lemmatization
                word = lemmatizer.lemmatize(word, 'v')  # eg. said
                if word == verb:  # said -> say, because of lemmatization
                    sentences.append(sent)

    return sentences


def lemmatize_graph(graph, tags):
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
            # eg. from "said" to "say"
            new_dict[k] = lemmatizer.lemmatize(word, 'v')
        else:
            new_dict[k] = word  # otherwise the normal word

    # update the dictionary with lemmatization
    return OurDependencyGraph(graph.graph, new_dict)


def hanks(verb):
    """
    Implementation of P. Hanks theory.
    Given a transitive verb, we find N sentences in the Brown corpus that
    contains the given verb. We do WSD (using 2 version of Lesk algorithm,
    one handwritten by us and the other from NLTK library) on the verb
    arguments (subj and obj), and finally, we compute the Filler's supersense
    incidence rate.
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
        tags = dict(nltk.pos_tag(tokens))  # dictionary of all PoS Tag of the tokens

        # Syntactic parsing
        result = dependency_parser.raw_parse(sentence)
        dep = next(result)
        graph = OurDependencyGraph()  # first init needed because of .init_from_dot()
        graph.init_from_dot(dep.to_dot())

        # Lemmatization
        # (it lemmatized only the verbs, the other words are not changed)
        lemmatized_graph = lemmatize_graph(graph, tags)  # es. "said" to "say"

        verb_key_list = lemmatized_graph.get_verb_key(verb)  # list of keys in which we can find the verb in graph.dict
        # format -> [int1, int 2, ...], eg.: [34], [0, 10, 34, ...]

        if len(verb_key_list) <= 0:
            # DEBUG
            # print("\tError in **{}**".format(sentence), file=sys.stderr)
            continue

        # Adjacency List
        # we take the first occurrence of the verb, which is our root
        adjs = lemmatized_graph.get_adj_neighbor(verb_key_list[0])
        # if the adjacent element of the verb are subj or obj we update adjs variable
        adjs = list(filter(lambda x: x[1] in subj_dept or x[1] in obj_dept, adjs))

        # Valency = 2
        if len(adjs) == 2:  # Note: not all the verb in sentences have valency = 2
            # assigning the correct subject and obj
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

    our_lesk_semantic_types = {}  # {(s1, s2): count}
    nltk_lesk_semantic_types = {}  # {(s1, s2): count}
    for f in fillers:
        # WSD

        # Our Lesk
        s1 = our_lesk(f[0], f[2])
        s2 = our_lesk(f[1], f[2])

        # nltk.wsd's Lesk
        s3 = lesk(f[2], f[0])
        s4 = lesk(f[2], f[1])

        if s1 is not None and s2 is not None:
            # Getting supersences
            t = (s1.lexname(), s2.lexname())

            # Getting frequency
            if t in our_lesk_semantic_types.keys():
                our_lesk_semantic_types[t] = our_lesk_semantic_types[t] + 1
            else:
                our_lesk_semantic_types[t] = 1

        if s3 is not None and s4 is not None:
            # Getting supersences
            t = (s3.lexname(), s4.lexname())

            # Getting frequency
            if t in nltk_lesk_semantic_types.keys():
                nltk_lesk_semantic_types[t] = nltk_lesk_semantic_types[t] + 1
            else:
                nltk_lesk_semantic_types[t] = 1

    print('\n[4.1] - "Our Lesk":\n\tFinding Semantic Clusters (percentage, count of instances, semantic cluster):')
    for key, value in sorted(our_lesk_semantic_types.items(), key=lambda x: x[1]):
        to_print = str(round((value / tot) * 100, 2))
        print("\t[{}%] - {} - {}".format(to_print, value, key))

    print('\n[4.2] - "NLTK Lesk":\n\tFinding Semantic Clusters (percentage, count of instances, semantic cluster):')
    for key, value in sorted(nltk_lesk_semantic_types.items(), key=lambda x: x[1]):
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
