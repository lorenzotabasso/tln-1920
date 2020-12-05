"""
Compute Concept Similarity.
Using three metrics, compute conceptual similarity on many couple of terms.
At the end, also computes the Pearson and Spearman correlation indexes.
"""

import time

from part2.exercise1.src.cs.WordNetAPIClient import WordNetAPIClient
from part2.exercise1.src.cs.correlation_indices import *
from part2.exercise1.src.cs.SimilarityMetrics import SimilarityMetrics


def parse_word_sim_353(path):
    """
    Support function, it parse the WordSim353 CSV file. Each line is compose by
    a couple of terms and their annotation.
    :param path: input path of the CSV file
    :return: a list rapresentation of the input file. Its format will be
    [(w1, w2, gold_annotation)]
    """
    result = []
    with open(path, 'r') as file:
        for line in file.readlines()[1:]:
            temp = line.split(",")
            gold_value = temp[2].replace('\n', '')
            result.append((temp[0], temp[1], float(gold_value)))

    return result


def conceptual_similarity(options):
    """
    Computes the conceptual similarity and writes the results in two CSV file
    in the output folder.

    :param options: a dictionary that contains the input and output paths.
    Format: { "input": "...", "output": "..." }
    """
    ws353 = parse_word_sim_353(options["input"])
    print("[1] - WordSim353.csv parsed.")

    wnac = WordNetAPIClient()

    similarities = []  # lista di liste di similarità, una lista per ogni metrica
    metric_obj = SimilarityMetrics(wnac)
    time_start = time.time()

    # A list of 2 tuples: the first containing the reference to the metrics
    # implementation, and the second containing his name (a tuple of 3 strings).
    metrics = list(zip(*metric_obj.get_all()))

    to_remove = []
    count_total_senses = 0  # to count the senses total

    # Looping over the list of all the three metrics.
    for metric in metrics[0]:
        sim_metric = []  # similarity list for this metric

        j = 0
        for couple_terms in ws353:
            synset1 = WordNetAPIClient.get_synsets(couple_terms[0])
            synset2 = WordNetAPIClient.get_synsets(couple_terms[1])
            # senses = [synset1, synset2]

            maxs = []  # list of senses similarity
            for s1 in synset1:
                for s2 in synset2:
                    count_total_senses += 1
                    maxs.append(metric(s1, s2))
            if len(maxs) == 0:  # word without senses (ex.: proper nouns)
                maxs = [-1]
                to_remove.append(j)
            sim_metric.append(max(maxs))
            j += 1
        similarities.append(sim_metric)

    time_end = time.time()
    print("[1] - Total senses similarity: {}".format(count_total_senses))
    print("[1] - Time elapsed: {0:0.2f} seconds".format(time_end - time_start))

    for j in range(len(ws353)):
        if j in to_remove:
            del ws353[j]
            for s in range(len(similarities)):
                del similarities[s][j]

    golden = [row[2] for row in ws353]  # the list of golden annotations

    pearson_list = []
    spearman_list = []

    for i in range(len(metrics[1])):
        yy = similarities[i]
        pearson_list.append(pearson_index(golden, yy))
        spearman_list.append(spearman_index(golden, yy))

    with open(options["output"] + 'task1_results.csv', "w") as out:
        out.write("word1, word2, {}, {}, {}, gold\n"
                  .format(metrics[1][0], metrics[1][1], metrics[1][2]))
        for j in range(len(ws353)):
            out.write("{0}, {1}, {2:.2f}, {3:.2f}, {4:.2f}, {5}\n"
                      .format(ws353[j][0], ws353[j][1], similarities[0][j],
                              similarities[1][j], similarities[2][j], ws353[j][2], )
                      )

    with open(options["output"] + 'task1_indices.csv', "w") as out:
        out.write(" , Pearson, Spearman\n")
        for j in range(len(pearson_list)):
            out.write("{}, {}, {}\n".format(metrics[1][j], str(pearson_list[j]), spearman_list[j]))
