import sys
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np
import requests
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def get_synset_terms(sense):
    """
    It use the BabelNet HTTP API for getting the first three Lemmas of the word
    associated to the given Babel Synset ID.
    :param sense: sense's BabelID
    :return: the first three lemmas of the given sense. An error string if
    there are none
    """

    url = "https://babelnet.io/v5/getSynset"
    params = {
        "id": sense,
        "key": "67adc825-bbcc-4cd4-8d6c-71ecdb875e7c",  # my API key
        "targetLang": "IT"  # Important: we are searching results in italian
    }

    req = requests.get(url=url, params=params)
    data = req.json()

    synset_terms = []

    i = 0  # used to loop over the first three terms
    j = 0  # used to loop over all the senses
    while j < len(data["senses"]) and i < 3:
        term = data["senses"][j]["properties"]["fullLemma"]

        if term not in synset_terms:
            synset_terms.append(term)
            i += 1
        j += 1

    if len(synset_terms) == 0:
        return "Empty synset terms"
    else:
        return synset_terms


def parse_nasari():
    """
    It parses the NASARI's Embedded input.
    :return: First: a dictionary in which each BabelID is associated with the
    corresponding NASARI's vector. Second: a lexical dictionary that associate
    to each BabelID the corresponding english term.

    {babelId: [nasari vector's values]}, {babelID: word_en}
    """

    nasari = {}
    babel_word_nasari = {}

    with open(options["nasari"], 'r', encoding="utf8") as file:
        for line in file.readlines():
            lineSplitted = line.split()
            k = lineSplitted[0].split("__")
            babel_word_nasari[k[0]] = k[1]
            lineSplitted[0] = k[0]
            i = 1
            values = []
            while i < len(lineSplitted):
                values.append(float(lineSplitted[i]))
                i += 1
            nasari[lineSplitted[0]] = values

    return nasari, babel_word_nasari


def parse_italian_synset():
    """
    It parses SemEvalIT17 file. Each italian term is associated with a list of
    BabelID.
    :return: a dictionary containing the italian word follower by the list of
    its BabelID. Format: {word_it: [BabelID]}
    """

    sem_dict = {}
    synsets = []
    term = "first_step"  # only for the first time
    with open(options["input_italian_synset"], 'r', encoding="utf8") as file:
        for line in file.readlines():
            line = line[:-1].lower()
            if "#" in line:
                line = line[1:]
                if term != "first_step":  # only for the first time
                    sem_dict[term] = synsets
                term = line
                synsets = []
            else:
                synsets.append(line)
    return sem_dict


def parse_input(path):
    """
    it parses the annotated words's file.
    :param path to the annotated word's file.
    :return: list of annotated terms. Format: [((w1, w2), value)]
    """

    annotation_list = []
    with open(path, 'r', encoding="utf-8-sig") as file:
        for line in file.readlines():
            splitted_line = line.split("\t")
            copule_words = (splitted_line[0].lower(), splitted_line[1].lower())
            value = splitted_line[2].replace("\n", "")
            annotation_list.append((copule_words, float(value)))
    return annotation_list


def similarity_vectors(babel_id_word1, babel_id_word2, nasari_dict):
    """
    It computes the cosine similarity between the two given NASARI vectors
    (with Embedded representation).
    :param babel_id_word1: list of BabelID of the first word
    :param babel_id_word2: list of BabelID of the second word
    :param nasari_dict: NASARI dictionary
    :return: the couple of senses (their BabelID) that maximise the score and
    the cosine similarity score.
    """

    max_value = 0
    senses = (None, None)
    for bid1 in babel_id_word1:
        for bid2 in babel_id_word2:
            if bid1 in nasari_dict.keys() and bid2 in nasari_dict.keys():
                # Storing the NASARI values of bid1 and bid2
                v1 = nasari_dict[bid1]
                v2 = nasari_dict[bid2]

                # Transforming the V1 and V2 array into a np.array (numpy array).
                # Array dimensions: 1 x len(v).
                n1 = np.array(v1).reshape(1, len(v1))
                n2 = np.array(v2).reshape(1, len(v2))

                # Computing and storing the cosine similarity.
                t = cosine_similarity(n1, n2)[0][0]
                if t > max_value:
                    max_value = t
                    senses = (bid1, bid2)
    return senses, max_value


def evaluate_correlation_level(v1, v2):
    """
    It evaluates the correlation between the system annotations (v2) and the
    human annotations (v1) using Pearson and Spearman  metrics.
    :param v1: the first annotated vector (human annotations)
    :param v2: the second annotated vector (algorithm annotations)
    :return: Pearson and Spearman indexes
    """

    return pearsonr(v1, v2)[0], spearmanr(v1, v2)[0]


def evaluate_annotation_agreement(v1, v2):
    """
    It compute the agrement level for every annotation inside v1 and v2.
    :param v1: the first annotated vector (human annotations)
    :param v2: the second annotated vector (algorithm annotations)
    :return: the Cohen Kappa index
    """

    return cohen_kappa_score(v1, v2)


global options  # Dictionary containing all the script settings. Used everywhere.

if __name__ == "__main__":

    options = {
        "input": "input/text-documents",
        "input_annotation": "input/my_words.txt",
        "input_italian_synset": "input/SemEval17_IT_senses2synsets.txt",
        "nasari": "input/mini_NASARI.tsv",
        "output": "output/"
    }

    nasari_dict, babel_word_nasari = parse_nasari()
    italian_senses_dict = parse_italian_synset()

    annotations = parse_input(options["input_annotation"])

    # Annotation's scores, used for evaluation
    scores_human = list(zip(*annotations))[1]

    annotations_algorithm = []
    for couple in annotations:
        key = couple[0]
        (s1, s2), score = similarity_vectors(italian_senses_dict[key[0]], italian_senses_dict[key[1]], nasari_dict)
        annotations_algorithm.append(((s1, s2), score))

    scores_algorithm = list(zip(*annotations))[1]

    # Task 1: Semantic Similarity
    pearson, spearman = evaluate_correlation_level(scores_human, scores_algorithm)
    print('Task 1: Semantic Similarity\n [1] - Person: {0}, Spearman: {1}'
          .format(pearson, spearman))

    # ------------------------------------------------------------------------------------------------------------------

    print("Computing semantic similarity. Please wait.\n")

    with open(options["output"] + 'results.txt', "w", encoding="utf-8") as out:

        # Progress bar
        progress_bar = tqdm(desc="Percentage", total=50, file=sys.stdout)

        for couple in annotations:
            key = couple[0]

            (s1, s2), score = similarity_vectors(italian_senses_dict[key[0]], italian_senses_dict[key[1]], nasari_dict)

            if s1 is not None and s2 is not None:
                out.write("{}\t{}\t{}\t{}\t".format(key[0], key[1], s1, s2))

                # print("{} - {}\n".format(key[0], key[1]))

                terms1 = get_synset_terms(s1)
                terms2 = get_synset_terms(s2)

                for g1 in terms1:
                    if g1 != terms1[len(terms1) - 1]:
                        out.write(g1 + ",")
                    else:
                        out.write(g1 + "\t")
                for g2 in terms2:
                    if g2 != terms2[len(terms2) - 1]:
                        out.write(g2 + ",")
                    else:
                        out.write(g2 + "\n")
            else:
                out.write("None\tNone\n")

            progress_bar.update(1)
