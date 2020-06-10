from nltk import parse
import re
import json
from anytree import AnyNode
from anytree.exporter import JsonExporter

from part1.src.utils import *


def parser(sentence, path_to_grammar):
    """
    Loads the NLTK's parser

    :param sentence: sentence to parse
    :param path_to_grammar: path to the grammar
    :return: all the parsing tree for the given sentence
    """
    parser = parse.load_parser(path_to_grammar)
    tokens = sentence.split()
    trees = parser.parse(tokens)
    return trees


def translate_word(word, is_proper_noun=False):
    """
    Function for 1:1 translate.

    :param word: word to translate
    :param is_proper_noun: flag, default False. If true, the word is a proper
    noun and it don't need translation.
    :return: the italian translation of the word
    """
    if not is_proper_noun:
        with open('../dictionary/dictionary_en_it.json') as json_file:
            lex = json.load(json_file)
        return lex[word]
    return word


def create_node(id, type, parent, label=None, proprieties=None):
    """
    Auxiliary method for the creation of tree's nodes. It encapsulate the
    Anynode function to to that.

    :param id: unique id of the node
    :param type: type of the node
    :param parent: parent node of the node to create
    :param label: translated word to insert in the node. Only present in leaves
    :param proprieties: node's features (dictionary of type {'tipoFeatures': 'valore'})
    :return: next available id. Node created.
    """
    id += 1
    if label is not None and proprieties is not None:
        return id, AnyNode(a=str(id), b=type, c=label, d=str(proprieties), parent=parent)
    if label is not None and proprieties is None:
        return id, AnyNode(a=str(id), b=type, c=label, parent=parent)
    return id, AnyNode(a=str(id), b=type, d=str(proprieties), parent=parent)


def sentence_1(tree):
    """
    Method than handles the first sentence: exists z1.(thing(z1) & image(you,z1)).
    """

    # Finding the verb. We know where is located thanks to his semantics -------
    terms = get_semantics(tree)

    term_verb = terms[1 + (len(terms) - 2)]  # When modifiers aren't present, terms's length is 2
    var_subj, variable_obj = get_arguments(term_verb)
    verb = match_pred_pos(tree, term_verb)

    # Finding the subject ------------------------------------------------------
    subj = match_pred_pos(tree, var_subj)

    # Finding the object -------------------------------------------------------
    occurrences_obj = find_occurrences(tree, variable_obj)
    obj = list(filter(lambda x: 'NNS' == x['tag'], occurrences_obj))[0]  # all the rest will be modifiers

    occurrences_obj.remove(obj)
    # the verb will appear in both Obj and Verb (because is a function of type f(x,y))
    if verb in occurrences_obj:
        occurrences_obj.remove(verb)

    # Building Sentence Plan tree
    id = 0
    id, root = create_node(id, type="clause", parent=None, label=None, proprieties={"tense": verb['tns']})

    # Subj
    id, tree_subj = create_node(id, type="subj", parent=root, label=translate_word(subj['pred']))

    # Verb
    id, tree_verb = create_node(id, type="verb", label=translate_word(verb['pred']), parent=root)

    # Object
    id, tree_object = create_node(id, type="obj", parent=root)
    id, tree_object_noun = create_node(id, type="noum", label=translate_word(obj['pred']),
                                       proprieties={"number": obj['num'], "gen": obj['gen']},
                                       parent=tree_object)

    # Taking all remaining modifiers (if there are any)
    for x in occurrences_obj:
        id, child = create_node(id, type="modifier", label=translate_word(x['pred']), parent=tree_object)

    return root


def sentence_2(tree):
    """
    Method than handles the second sentence: exists x.(exists e.(presence(e) & agent(e,x)) &
    exists z2.(my(z2) & head(z2) & exists z8.(price(z8) & x(z8)) & on(x,z2)))
    """

    visited_variables = set()  # used below in order to speedup

    # Finding the verb ---------------------------------------------------------
    terms = get_semantics(tree)
    var_verb = terms[0]  # presence(e)
    verb = match_pred_pos(tree, var_verb)  # {'pred': 'presence', 'tag': 'VBZ', 'num': 'sg', 'tns': 'pres'}
    var_event = str(var_verb.args[0])  # e
    visited_variables.add(var_event)

    # Finding the subject ------------------------------------------------------
    var_subj = get_intransitive_subject(tree)  # x
    visited_variables.add(var_subj)

    # if there are any subject modifiers
    mod_subj = find_occurrences(tree, var_subj)  # [{'pred': 'on', 'tag': 'IN', 'loc': True}]

    # filtering only the INs
    subj = list(filter(lambda x: 'IN' == x['tag'], mod_subj))[0]
    # subj = {'pred': 'on', 'tag': 'IN', 'loc': True}
    mod_subj.remove(subj)  # []

    # Finding the object, only in the unvisited variables ----------------------
    var_compl = get_all_variables(tree) - visited_variables  # {'z8', 'z2'}

    # z8 occurrences:
    # {'pred': 'price', 'tag': 'NN', 'num': 'sg', 'gen': 'm'}

    # z2 occurrences:
    # [ {'pred': 'my', 'tag': 'PRPS', 'num': 'sg'},
    # {'pred': 'head', 'tag': 'NN', 'num': 'sg', 'gen': 'f'},
    # {'pred': 'on', 'tag': 'IN', 'loc': True} ]

    occurrences_compl = []  # object's occurrences
    for variable in var_compl:
        for occurrence in find_occurrences(tree, variable):
            occurrences_compl.append(occurrence)
            # the obj will appear in both Subj and Obj (because is a function
            # of type f(x,y)). Mod_subj is already empty, but if it appears
            # again (for instance with another input sentence) we have to
            # remove it from subj.
            if occurrence in mod_subj:
                mod_subj.remove(occurrence)

    # Building Sentence Plan tree ----------------------------------------------
    id = 0
    id, root = create_node(id, type="clause", parent=None, proprieties={"tense": verb['tns']})

    # Verb
    id, tree_verb = create_node(id, type="verb", parent=root, label=translate_word(verb['pred']))

    # Subject
    id, tree_subj = create_node(id, type="subj", parent=root)
    id, tree_subj_specifier = create_node(id, type="spec", parent=tree_subj, label="un")
    id, tree_subj_noun = create_node(id, type="noum", parent=tree_subj, label=translate_word(var_subj))
    for x in mod_subj:
        id, child = create_node(id, type="modifier", parent=tree_subj, label=translate_word(x['pred']))

    # Object complement
    id, tree_object_complement = create_node(id, type="complement", parent=root)

    # Searching for preposition ("on")
    for x in occurrences_compl:
        if x['tag'] == 'IN':
            id, child = create_node(id, type="prep", parent=tree_object_complement, label=translate_word(x['pred']))
            occurrences_compl.remove(x)

    # Object complement with preposition
    id, child_np = create_node(id, type="ppcompl", parent=tree_object_complement)

    for x in occurrences_compl:
        if x['tag'] == 'PRPS':
            id, child = create_node(id, type="modifier", parent=child_np, label=translate_word(x['pred']))
            occurrences_compl.remove(x)

    for x in occurrences_compl:
        if x['tag'] == 'NN' and x['pred'] == 'head':
            id, child = create_node(id, type="noum", parent=child_np, label=translate_word(x['pred']),
                                    proprieties={"num": x['num'], "gen": x['gen']})
            occurrences_compl.remove(x)

    return root


def sentence_3(tree):
    """
    Method than handles the third sentence: exists x.(your(x) & big(x) & opportunity(x) & exists e.(fly(e) & agent(e,
    x) & out(e) & exists y.(from(e,y) & here(y))))
    """

    visited_variables = set() # used below in order to speedup

    # Finding the verb ---------------------------------------------------------
    terms = get_semantics(tree)
    var_verb = terms[1 + (len(terms) - 6)]  # when there are not present modifiers, the list has length 6
    var_event = str(var_verb.args[0])  # "e"
    mod_verb = find_occurrences(tree, var_event)  # {from, out, fly (removed below)}
    visited_variables.add(var_event)  # {"e"}
    verb = match_pred_pos(tree, var_verb)  # fly(e)
    mod_verb.remove(verb)  # Removing the verb from the modifiers, because "e" is the verb argument (fly(e))

    # Finding the subject ------------------------------------------------------
    var_subj = get_intransitive_subject(tree)  # "x"
    visited_variables.add(var_subj)  # {"e", "x"}
    mod_subj = find_occurrences(tree, var_subj)  # {opportunity, big, your}
    subj = list(filter(lambda x: 'NN' == x['tag'], mod_subj))[0]  # opportunity(x)
    mod_subj.remove(subj)  # Removing the subj from the modifiers, because "x" is the subj argument (opportunity(x))

    # Finding the object, only in the unvisited variables ----------------------
    var_compl = get_all_variables(tree) - visited_variables  # { "y" }
    occurrences_compl = []  # complement occurrences { from, here }
    for variable in var_compl:
        for occ in find_occurrences(tree, variable):
            occurrences_compl.append(occ)
            # the obj will appear in both Verb and Obj (because is a function
            # of type f(x,y)). So, we need to remove it from Verb
            if occ in mod_verb:
                mod_verb.remove(occ)  # we will remove only "from"

    # Building Sentence Plan tree ----------------------------------------------
    id = 0
    id, root = create_node(id, type="clause", parent=None, proprieties={"tense": verb['tns']})

    # Subject
    id, tree_subj = create_node(id, type="subj", parent=root)
    id, tree_subj_specifier = create_node(id, type="spec", parent=tree_subj, label="la")

    # Adding all subjects modifier
    for mod in mod_subj:
        id, tree_subj_modifier = create_node(id, type="modifier", parent=tree_subj, label=translate_word(mod['pred']))

    id, tree_subj_noun = create_node(id, type="noum", parent=tree_subj, label=translate_word(subj['pred']),
                             proprieties={"gen": subj['gen']})

    # Verb and Adverbs
    id, tree_verb = create_node(id, type="verb", parent=root)
    id, tree_main_verb = create_node(id, type="vrb", parent=tree_verb, label=translate_word(verb['pred']))
    for adv in mod_verb:
        id, tree_verb_adverb = create_node(id, type="adv", parent=tree_verb, label=translate_word(adv['pred']))

    # Place complement with its Adverbs
    id, tree_place_compl = create_node(id, type="complement", parent=root)

    for prep in occurrences_compl:
        if prep['tag'] == 'IN':  # from
            id, child = create_node(id, type="prep", parent=tree_place_compl, label=translate_word(prep['pred']))
            occurrences_compl.remove(prep)

    for adv in occurrences_compl:
        if adv['tag'] == 'RB':  # here
            if 'loc' in adv.keys():
                id, ppcompl = create_node(id, type="ppcompl", parent=tree_place_compl,
                                          label=translate_word(adv['pred']))
            occurrences_compl.remove(adv)

    return root


if __name__ == "__main__":

    grammar_path = "../grammars/my-simple-sem.fcfg"

    sentences = ["you are imagining things",
                 "there is a price on my head",
                 "your big opportunity is flying out of here"]

    # Using regex to filter the correct semantic output
    correct_regex = [
        'exists\\s\\w+.\\(\\w+\\(\\w+\\)\\s\\&\\s(\\w+\\(\\w+\\)\\s&\\s)*\\w+\\(\\w+,\\w+\\)\\)',
        'exists\\s\w+.\\(exists\\s\\w+.\\(\\w+\\(\\w+\\)\\s&\\s\\w+\\(\\w+,\\w+\\)\\)\\s&\\sexists\\s\\w+.\\(\\w+\\('
        '\\w+\\)\\s&\\s\\w+\\(\\w+\\)\\s&\\sexists\\s\\w+.\\(\\w+\\(\\w+\\)\\s&\\s\\w+\\(\\w+\\)\\)\\s&\\s\\w+\\('
        '\\w+,\\w+\\)\\)\\)',
        'exists\\s\\w+.\\(your\\(\\w+\\)\\s&\\sbig\\(\\w+\\)\\s&\\sopportunity\\(\\w+\\)\\s&\\sexists\\s\\w+.\\('
        'fly\\(\\w+\\)\\s&\\sagent\\(\\w+,\\w+\\)\\s&\\sout\\(\\w+\\)\\s&\\sexists\\s\\w+.\\(from\\(\\w+,'
        '\\w+\\)\\s&\\shere\\(\\w+\\)\\)\\)\\)'
    ]

    # Overwrite sentences array with a more convenient map
    sentences = list(map(lambda x: x.lower(), sentences))

    for index in range(len(sentences)):
        extracted_sentence = sentences[index]
        correct_tree = None
        all_trees = parser(extracted_sentence, grammar_path)

        tree = best_tree(all_trees)
        semantic = str(tree.label()['SEM'])

        k = 0
        for i in range(len(correct_regex)):
            if re.match(correct_regex[i], semantic):
                k = i
                break

        print("Match Found! RegExpr number {}.\n{}\n".format(str(k+1), semantic))

        root = None
        if k == 0:
            root = sentence_1(tree)
        elif k == 1:
            root = sentence_2(tree)
        elif k == 2:
            root = sentence_3(tree)

        exporter = JsonExporter(indent=2, sort_keys=True)
        with open('../output/' + 'sentence_plan_' + str(index) + '.json', 'w') as file:
            exporter.write(root, file)
