import re
import csv
import nltk
from nltk.corpus import framenet as fn

from part2.exercise2.src.WordNetAPIClient import WordNetAPIClient


def get_main_clause(frame_name):
    """
    Get of the main clause from the frame name (in italian "reggente").
    :param frame_name: the name of the frame
    :return: the main clause inside the frame name
    """
    tokens = nltk.word_tokenize(re.sub('\_', ' ', frame_name))
    tokens = nltk.pos_tag(tokens)

    for elem in reversed(tokens):
        if elem[1] == "NN" or elem[1] == "NNS":
            return elem[0]


def populate_contexts(frame, mode):
    """
    It populates 2 disambiguation context (one for Framenet and onw for Wordnet)
    given a frame name.

    :param frame: the frame name.
    :param mode: a string indicating the way to create context the possibility
    are: "Frame name", "FEs" and "LUs".
    :return: two list (ctx_f, ctx_w) representing the populated contexts.
    """
    ctx_f = []  # the Framenet context
    ctx_w = {}  # the Wordnet context
    wnac = WordNetAPIClient()

    if mode == "Frame name":
        # The context in this case contains the frame name and his definition.
        ctx_f = [get_main_clause(frame.name), frame.definition]

        # Here, the context is a list of synset associated to the frame name.
        # In each synset are usually present word, glosses and examples.
        ctx_w = wnac.get_disambiguation_context(get_main_clause(frame.name))

    elif mode == "FEs":
        # Populating ctx_w for FEs
        for key in frame.FE:
            ctx_f.append(key)
            ctx_f.append(frame.FE[key].definition)

            # copying all the values inside the ctx_w dictionary.
            temp = wnac.get_disambiguation_context(key)
            for k in temp:
                ctx_w[k] = temp[k]

    elif mode == "LUs":
        # Populating ctx_w for LUs
        for key in frame.lexUnit:
            lu_key = re.sub('\.[a-z]+', '', key)
            ctx_f.append(lu_key)
            # ctx_f.append(frame.lexUnit[key].definition)

            # copying all the values inside the ctx_w dictionary.
            temp = wnac.get_disambiguation_context(lu_key)
            for k in temp:
                ctx_w[k] = temp[k]

    return ctx_f, ctx_w


def bag_of_words(ctx_fn, ctx_wn):
    """
    Given two disambiguation context, it returns the bag of words mapping
    between the input arguments.
    :param ctx_fn: the first disambiguation context (from Framenet)
    :param ctx_wn: the second disambiguation context (from Wordnet)
    :return: the synset with the highest score
    """
    sentences_fn = set()  # set of all Framenet FEs and their descriptions
    sentences_wn = {}  # dictionary of all Wordnet sysnset, glosses and examples.
    ret = None  # temporary return variable
    temp_max = 0  # temporary variable for the score

    for sentence in ctx_fn:
        for word in sentence.split():
            word_clean = re.sub('[^A-Za-z0-9 ]+', '', word)
            sentences_fn.add(word_clean)

    # transform the ctx_w dictionary into a set, in order to compute
    # intersection.
    for key in ctx_wn:  # for each WN synset
        temp_set = set()
        for sentence in ctx_wn[key]:  # for each sentence inside WN synset
            if sentence:
                for word in sentence.split():
                    temp_set.add(word)  # add words to temp_set

        # computing intersection between temp_set and sentences_fn.
        # Putting the result inside sentences_wn[key].
        # Each entry in sentences_wn will have the cardinality of the
        # intersection as his "score" at the first position.
        sentences_wn[key] = (len(temp_set.intersection(sentences_fn)), temp_set)

        # update max score and save the associated sentence.
        if temp_max < sentences_wn[key][0]:
            temp_max = sentences_wn[key][0]
            ret = (key, sentences_wn[key])

    if ret:
        return ret[0]  # return the synset with the highest score
    else:
        ""


def evaluate():
    """
    Doing output evaluation and print the result on the console.
    """
    total_len = 0
    test = 0
    with open(config["output"] + 'results.csv', "r", encoding="utf-8") as results:
        with open(config["golden"], "r", encoding="utf-8") as golden:
            reader_results = csv.reader(results, delimiter=',')
            reader_golden = csv.reader(golden, delimiter=',')

            items_in_results = []  # list of items in results
            items_in_golden = []  # list of items in gold

            for line_out in reader_results:
                items_in_results.append(line_out[-1])

            for line_golden in reader_golden:
                items_in_golden.append(line_golden[-1])

            # counting equal elements
            i = 0
            while i < len(items_in_results):
                if items_in_results[i] == items_in_golden[i]:
                    test += 1
                i += 1

            total_len = i

    print("\nPrecision: {0} / {1} Synsets -> {2:.2f} %".format(test, total_len, (test / total_len) * 100))


global config  # Dictionary containing all the script settings. Used everywhere.

if __name__ == "__main__":

    config = {
        "output": "/Users/lorenzotabasso/Desktop/University/TLN/Progetto/19-20/tln-1920/part2/exercise2/output/",
        "golden": "/Users/lorenzotabasso/Desktop/University/TLN/Progetto/19-20/tln-1920/part2/exercise2/input/gold.csv"
    }

    # getFrameSetForStudent('Tabasso')
    # student: Tabasso
    # 	ID:  133	frame: Process_start
    # 	ID: 2980	frame: Transition_to_a_situation
    # 	ID:  405	frame: Performing_arts
    # 	ID: 1927	frame: Scope
    # 	ID: 2590	frame: Business_closure

    frame_ids = [133, 2980, 405, 1927, 2590]

    with open(config["output"] + 'results.csv', "w", encoding="utf-8") as out:

        print("Assigning Synsets...")

        for frame in frame_ids:
            f = fn.frame_by_id(frame)

            ctx_f, ctx_w = populate_contexts(f, "Frame name")
            sense_name = bag_of_words(ctx_f, ctx_w)

            out.write("Frame name, {0}, Wordnet Synset, {1}\n".format(f.name, sense_name))

            ctx_f, ctx_w = populate_contexts(f, "FEs")
            i = 0
            while i < len(ctx_f) - 2:
                fe = [ctx_f[i], ctx_f[i + 1]]
                sense_fes = bag_of_words(fe, ctx_w)
                out.write("Frame elements, {0}, Wordnet Synset, {1}\n".format(fe[0], sense_fes))
                i += 2

            ctx_f, ctx_w = populate_contexts(f, "LUs")
            for lu in ctx_f:
                sense_lus = bag_of_words(lu, ctx_w)
                out.write("Frame lexical unit, {0}, Wordnet Synset, {1}\n".format(lu, sense_lus))

        print("Done. Starting evaluation.")

    evaluate()
