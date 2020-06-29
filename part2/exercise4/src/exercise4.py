import re
import sys

from numpy import mean
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np
import requests
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity


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

        # added some preprocess
        term = re.sub('\_', ' ', term).lower()

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


def parse_word(path):
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


def parse_sense(path):
    """
    it parses the senses in the file.
    :param path to the senses file.
    :return: list of annotated senses and associated terms. Format: [(s1, s2, t1, t2)]
    """

    sense_list = []
    with open(path, 'r', encoding="utf-8-sig") as file:
        for line in file.readlines():
            # print(line)
            splitted_line = line.split("\t")
            couple_word = (splitted_line[0].lower(), splitted_line[1].lower())
            copule_sense = (splitted_line[2].lower(), splitted_line[3].lower())
            couple_terms = (splitted_line[4].lower(), splitted_line[5].lower().replace("\n", ""))
            sense_list.append((couple_word[0], couple_word[1], copule_sense[0], copule_sense[1], couple_terms[0], couple_terms[1]))
    return sense_list


def similarity_vector(babel_id_word1, babel_id_word2, nasari_dict):
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


global options  # Dictionary containing all the script settings. Used everywhere.

if __name__ == "__main__":

    options = {
        "input_annotation_1": "input/mydata/my_words.txt",
        "input_annotation_2": "input/mydata/my_words_2.txt",
        "input_senses": "input/mydata/my_senses.tsv",
        "input_italian_synset": "input/SemEval17_IT_senses2synsets.txt",
        "nasari": "input/mini_NASARI.tsv",
        "output": "output/"
    }

    nasari_dict, babel_word_nasari = parse_nasari()
    italian_senses_dict = parse_italian_synset()

    # Task 1: Semantic Similarity
    #
    # 1. annotate by hand the couple of words in [0,4] range
    # 2. compute inter-rate agreement with Spearman and Pearson indexes
    # 3. Compute the cosine similarity between the hand-annotated scores and
    # Nasari best score given the two terms
    # 4. Evaluate the total quality using again the Spearman and Pearson
    # indexes between the human annotation scores and the Nasari scores.

    print('Task 1: Semantic Similarity')

    annotations_1 = parse_word(options["input_annotation_1"])
    annotations_2 = parse_word(options["input_annotation_2"])

    # Annotation's scores, used for evaluation
    scores_human_1 = list(zip(*annotations_1))[1]
    scores_human_2 = list(zip(*annotations_2))[1]

    # Computing the mean value for each couple of annotation score
    scores_human_mean = [(x + y) / 2 for x, y in zip(scores_human_1, scores_human_2)]
    print('\tMean value: {0:.2f}'.format(mean(scores_human_mean)))

    # 2. Computing the inter-rate agreement. This express if the two annotations are consistent
    inter_rate_pearson, inter_rate_spearman = evaluate_correlation_level(scores_human_1, scores_human_2)
    print('\tInter-rate agreement - Pearson: {0:.2f}, Spearman: {1:.2f}'
          .format(inter_rate_pearson, inter_rate_spearman))

    # 3. Computing the cosine similarity between the hand-annotated scores and
    # Nasari best score given the two terms
    annotations_algorithm = []
    for couple in annotations_1:  # is equal to use annotations_1 or annotations_2, because the words are the same
        key = couple[0]
        (s1, s2), score = similarity_vector(italian_senses_dict[key[0]], italian_senses_dict[key[1]], nasari_dict)
        annotations_algorithm.append(((s1, s2), score))

    scores_algorithm = list(zip(*annotations_algorithm))[1]

    # 4. Evaluate the total quality using again the Spearman and Pearson
    # indexes between the human annotation scores and the Nasari scores.
    pearson, spearman = evaluate_correlation_level(scores_human_mean, scores_algorithm)
    print('\tEvaluation - Person: {0:.2f}, Spearman: {1:.2f}'.format(pearson, spearman))

    # ------------------------------------------------------------------------------------------------------------------

    print("\nTask 2: Sense Identification.")
    # Task 2: Sense Identification
    #
    # 1. annotate by hand the couple of words in the format specified in the README
    # 2. compute inter-rate agreement with the Cohen's Kappa score
    # 3. Compute the cosine similarity between the hand-annotated scores and
    # Nasari best score given the two terms
    # 4. Evaluate the total quality using the argmax function. Evaluate both
    # the single sense and both the senses in the couple.

    int_score_human_1 = [int(x) for x in scores_human_1]
    int_score_human_2 = [int(x) for x in scores_human_2]

    # 2. Computing the inter-rate agreement. This express if the two score are consistent
    k = cohen_kappa_score(int_score_human_1, int_score_human_2)
    print('\tInter-rate agreement - Cohen Kappa: {0:.2f}'.format(k))

    senses = parse_sense(options["input_senses"])

    with open(options["output"] + 'results.tsv', "w", encoding="utf-8") as out:

        i = 0  # used for print progress bar
        first_print = True # used for print progress bar

        # used for final comparison. It is a in-memory copy of the output file
        nasari_out = []
        for row in senses:

            # In this case I re-use the similarity_vector function, which use
            # the cosine similarity to compute again the two senses that
            # produce the maximal similarity score. The "score" variable is
            # unused, so it's substituted by the don't care variable "_".
            (s1, s2), _ = similarity_vector(italian_senses_dict[row[0]], italian_senses_dict[row[1]], nasari_dict)

            # if both Babel Synset exists and are not None
            if s1 is not None and s2 is not None:
                out.write("{}\t{}\t{}\t{}\t".format(row[0], row[1], s1, s2))

                out_terms_1 = get_synset_terms(s1)
                out_terms_2 = get_synset_terms(s2)
                nasari_terms_1 = ""
                nasari_terms_2 = ""

                for t1 in out_terms_1:
                    if t1 != out_terms_1[len(out_terms_1) - 1]:
                        out.write(t1 + ",")  # if not the last term, put a ","
                        nasari_terms_1 += t1 + ","
                    else:
                        out.write(t1 + "\t")  # otherwise put a separator
                        nasari_terms_1 += t1

                for t2 in out_terms_2:
                    if t2 != out_terms_2[len(out_terms_2) - 1]:
                        out.write(t2 + ",")  # if not the last term, put a ","
                        nasari_terms_2 += t2 + ","
                    else:
                        out.write(t2 + "\n")  # otherwise put a separator
                        nasari_terms_2 += t2
            else:
                out.write("{}\t{}\tNone\tNone\tNone\tNone\n".format(row[0], row[1]))

            # updating percentage
            i += 1

            if first_print:
                print('\tDownloading terms from BabelNet.')
                print('\t#', end="")
                first_print = False
            if i % 10 == 0:
                print('#', end="")
            else:
                print('-', end="")

            # populate the nasari_out list.
            nasari_out.append((row[0], row[1], s1, s2, nasari_terms_1, nasari_terms_2))

        count_single = 0
        count_couple = 0
        for sense_row in senses:
            for nasari_row in nasari_out:
                arg0 = sense_row[2] == nasari_row[2]
                arg1 = sense_row[3] == nasari_row[3]
                if arg0:
                    count_single += 1
                if arg1:
                    count_single += 1
                if arg0 and arg1:
                    count_couple += 1
        print("\n\tSingle: {0} / 100 ({0}%) - Couple: {1} / 50 ({2:.0f}%)"
              .format(count_single, count_couple, (count_couple * 100 / 50)))
