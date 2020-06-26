import re
import sys
import csv

from nltk.corpus import framenet as fn
import hashlib
from random import randint
from random import seed
import nltk
from tqdm import tqdm

from part2.exercise2.src.WordNetAPIClient import WordNetAPIClient


def print_frames_with_ids():
    for x in fn.frames():
        print('{}\t{}'.format(x.ID, x.name))


def get_frams_ids():
    return [f.ID for f in fn.frames()]


def get_frame_set_for_student(surname, list_len=5):
    nof_frames = len(fn.frames())
    base_idx = (abs(int(hashlib.sha512(surname.encode('utf-8')).hexdigest(), 16)) % nof_frames)
    print('\nstudent: ' + surname)
    framenet_IDs = get_frams_ids()
    i = 0
    offset = 0
    seed(1)
    while i < list_len:
        fID = framenet_IDs[(base_idx + offset) % nof_frames]
        f = fn.frame(fID)
        fNAME = f.name
        print('\tID: {a:4d}\tframe: {framename}'.format(a=fID, framename=fNAME))
        offset = randint(0, nof_frames)
        i += 1


# Printing frames information.
def print_frame(frameId):
    frame = fn.frame(frameId)
    print("FRAME ID, NAME: {}, {}".format(frame.ID, frame.name))
    print("{0}\n{1}".format(frame.definition, frame.FE))
    #     print('\n****************************\n\n')
    #     print("FRAME ELEMENTS: \n {}".format(frame.FE))
    #     print('\n****************************\n\n')
    #     print("Lexical Units: \n {}".format(frame.lexUnit))
    #     print('\n_________________________________________________________\n\n')

    print_frame(133)
    # print_frame(2980)
    # print_frame(405)
    # print_frame(1927)
    # print_frame(2590)  # Business_closure: chiusura di business


# get del reggente
def get_main_clause(frame_name):
    tokens = nltk.word_tokenize(re.sub('\_', ' ', frame_name))
    tokens = nltk.pos_tag(tokens)

    for elem in reversed(tokens):
        if elem[1] == "NN" or elem[1] == "NNS":
            return elem[0]


def populate_contexts(frame, mode):
    ctx_f = []
    ctx_w = []
    wnac = WordNetAPIClient()

    if mode == "Frame name":
        ctx_f = [get_main_clause(frame.name), frame.definition]
        ctx_w = wnac.get_disambiguation_context(get_main_clause(frame.name))

    elif mode == "FEs":
        fe_key = ""

        # Populating ctx_w for FEs
        for key in frame.FE:
            fe_key = key
            ctx_f.append(key)
            ctx_f.append(frame.FE[key].definition)

        ctx_w = wnac.get_disambiguation_context(fe_key)

    elif mode == "LUs":
        lu_key = ""

        # Populating ctx_w for LUs
        for key in frame.lexUnit:
            lu_key = re.sub('\.[a-z]+', '', key)
            ctx_f.append(lu_key)
            # ctx_f.append(frame.lexUnit[key].definition)

        ctx_w = wnac.get_disambiguation_context(lu_key)

    return ctx_f, ctx_w


def mapping(ctx_fn, ctx_wn, mode):
    if mode == "Frame name":

        sentences_fn = set()
        for sentence in ctx_fn:
            for word in sentence.split():
                word_clean = re.sub('[^A-Za-z0-9 ]+', '', word)
                sentences_fn.add(word_clean)

        sentences_wn = {}
        ret = None
        temp_max = 0
        for key in ctx_wn:
            temp_set = set()
            for sentence in ctx_wn[key]:
                if sentence:
                    for word in sentence.split():
                        temp_set.add(word)
            sentences_wn[key] = (len(temp_set.intersection(sentences_fn)), temp_set)

            if temp_max < sentences_wn[key][0]:
                temp_max = sentences_wn[key][0]
                ret = (key, sentences_wn[key])

        # print("Max:{}".format(ret))
        return ret[0]

    elif mode == "FEs" or mode == "LUs":
        sentences_fn = set()
        for sentence in ctx_fn:
            for word in sentence.split():
                word_clean = re.sub('[^A-Za-z0-9 ]+', '', word)
                sentences_fn.add(word_clean)

        sentences_wn = {}
        ret = None
        temp_max = 0
        for key in ctx_wn:
            temp_set = set()
            for sentence in ctx_wn[key]:
                if sentence:
                    for word in sentence.split():
                        temp_set.add(word)
            sentences_wn[key] = (len(temp_set.intersection(sentences_fn)), temp_set)

            if temp_max < sentences_wn[key][0]:
                temp_max = sentences_wn[key][0]
                ret = (key, sentences_wn[key])

        # print("Max:{}".format(ret))
        if ret:
            return ret[0]
        else:
            list()


def evaluate():
    total_len = 0
    test = 0
    with open(options["output"] + 'results.csv', "r", encoding="utf-8") as results:
        with open(options["golden"], "r", encoding="utf-8") as golden:
            reader_out = csv.reader(results, delimiter=',')
            reader_golden = csv.reader(golden, delimiter=',')

            items_in_out = []
            items_in_golden = []

            for line_out in reader_out:
                items_in_out.append(line_out[-1])

            for line_golden in reader_golden:
                items_in_golden.append(line_golden[-1])

            i = 0
            while i < len(items_in_out):
                if items_in_out[i] == items_in_golden[i]:
                    test += 1
                i += 1

            total_len = i

    print("\nPrecision: {0} / {1} Synsets -> {2:.2f} %".format(test, total_len, (test/total_len)*100))


global options  # Dictionary containing all the script settings. Used everywhere.

if __name__ == "__main__":

    options = {
        "output": "/Users/lorenzotabasso/Desktop/University/TLN/Progetto/19-20/tln-1920/part2/exercise2/output/",
        "golden": "/Users/lorenzotabasso/Desktop/University/TLN/Progetto/19-20/tln-1920/part2/exercise2/output/golden.csv"
    }

    # getFrameSetForStudent('Tabasso')

    # student: Tabasso
    # 	ID:  133	frame: Process_start
    # 	ID: 2980	frame: Transition_to_a_situation
    # 	ID:  405	frame: Performing_arts
    # 	ID: 1927	frame: Scope
    # 	ID: 2590	frame: Business_closure

    frame_ids = [133, 2980, 405, 1927, 2590]

    with open(options["output"] + 'results.csv', "w", encoding="utf-8") as out:

        # Progress bar
        # progress_bar = tqdm(desc="Percentage", total=5, file=sys.stdout)
        print("Assigning Synsets...")

        for frame_id in frame_ids:
            f = fn.frame_by_id(frame_id)

            ctx_f, ctx_w = populate_contexts(f, "Frame name")
            sense_name = mapping(ctx_f, ctx_w, "Frame name")

            out.write("Frame name, {0}, Wordnet Synset, {1}\n".format(f.name, sense_name))

            ctx_f, ctx_w = populate_contexts(f, "FEs")
            i = 0
            while i < len(ctx_f) - 2:
                fe = [ctx_f[i], ctx_f[i + 1]]
                sense_fes = mapping(fe, ctx_w, "FEs")
                out.write("Frame elements, {0}, Wordnet Synset, {1}\n".format(fe[0], sense_fes))
                i += 2

            ctx_f, ctx_w = populate_contexts(f, "LUs")
            for lu in ctx_f:
                sense_lus = mapping(lu, ctx_w, "LUs")
                out.write("Frame lexical unit, {0}, Wordnet Synset, {1}\n".format(lu, sense_lus))

            # progress_bar.update(1)
        print("Done. Starting evaluation.")

    evaluate()
