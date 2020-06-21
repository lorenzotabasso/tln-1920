import re


def best_tree(trees):
    """
    Given a list of trees, it returns the correct tree, the one with no
    lambda-expression inside.

    :param trees: list of trees
    :return: the correct tree
    """

    final_tree = None
    for tree in trees:
        flag = True
        tree_semantic = tree.label()['SEM']
        string_semantic = str(tree_semantic)
        string_split = string_semantic.split()
        for character in string_split:
            if "\\" in character:
                flag = False

        if flag:
            final_tree = tree

    return final_tree


def get_all_variables(tree):
    """
    It returns a set containing all the variables inside the given tree.

    :param tree: the given tree
    :return: set of tree's variables
    """
    terms = list(subterms(tree.label()['SEM'].term))
    variables = set()
    for term in terms:
        for a in term.args:
            variables.add(a.variable.name)

    return variables


def get_semantics(tree):
    """
    It returns the semantics for the given tree.

    :param tree: input tree
    :return: all the terms of the semantics
    """
    return subterms(tree.label()['SEM'].term)


def get_arguments(node):
    """
    It returns the argument inside a node.

    :param node: the tree node to explore
    :return: a list containing the variable name of the node and all the other
    args inside the node
    """
    return list(map(lambda x: x.variable.name, node.args))


def find_occurrences(tree, variable):
    """
    It returns all the occurrences inside a semantics of the given variable
    (in all the tree).

    :param tree: the tree in which perform the search
    :param variable: variable to find
    :return: a list of all leaves containing the given variable
    """
    res = []
    terms = aux_find_semantic_occurrences(tree, variable)
    for term in terms:
        node = match_pred_pos(tree, term)
        if node is not None and not res.__contains__(node):
            res.append(node)

    return res


def aux_find_semantic_occurrences(tree, variable):
    """
    Auxiliary function of find_occurrences. It explroes the given tree and it
    returns a list with all the nodes that contains the given variable.

    :param tree: the tree in which perform the search
    :param variable: the variable to find
    :return: a list of all leaves containing the given variable
    """

    result = set()
    term = tree.label()['SEM'].term
    terms = subterms(term)
    for term in terms:
        # We first return all term's arguments
        term_args = list(map(lambda x: x.variable.name, term.args))
        if variable in term_args:
            result.add(term)
    return list(result)


def get_intransitive_subject(tree):
    """
    :param tree: semantic tree
    :return: the variable corresponding to the subject. For example, given the
        following semantic tree:

        exists x.(exists e.(presence(e) & agent(e,x)) & exists z2.(my(z2) &
        head(z2) & exists z8.(price(z8) & x(z8)) & on(x,z2)))

        which is the semantic tree of the second sentence, it returns "x"
    """
    terms = subterms(tree.label()['SEM'].term)
    agent = list(filter(lambda x: x.pred.variable.name == 'agent', terms))[0]
    subj = agent.args[1].variable.name
    return subj


def subterms(superterm):
    """
    It Returns a list of all the subterms inside the superterm.

    :param superterm: the "super" (the upper class) term to divide in subterms
    :return: a list of all the terms inside superterm
    """
    terms = []
    aux_subterms(superterm, terms)
    return terms


def aux_subterms(superterm, terms):
    # for differentiate the tree composition
    if hasattr(superterm, 'term'):
        aux_subterms(superterm.term, terms)
    if hasattr(superterm, 'first'):
        aux_subterms(superterm.first, terms)
    if hasattr(superterm, 'second'):
        aux_subterms(superterm.second, terms)
    elif hasattr(superterm, 'pred'):
        if not terms.__contains__(superterm):  # maintains the sort order
            terms.append(superterm)  # leaf


def match_pred_pos(tree, term):
    """
    It return the match between the syntattic term in input and its PoS-Tag.

    :param tree: the semantic tree to explore
    :param term: the semantic term to find the PoS-Tag
    :return: a node with the PoS-Tag information
    """
    leaves = ['DT', 'EX', 'IN', 'JJ', 'NN', 'NNS', 'PRP', 'PRPS', 'RB', 'VBG', 'VBP', 'VBZ']
    pred_name = term.pred.variable.name if hasattr(term, 'pred') else term

    # Returns al the possible subtrees
    all_subtrees = [subtree for subtree in tree.subtrees()]

    # Filtering all the subtrees. After the filter(), in all_subtrees there will be only the leaves
    all_subtrees = list(
        filter(lambda x: re.search("\\'(.*)\\'", str(x.label()).split('\n')[0], re.IGNORECASE).group(1) in leaves,
               all_subtrees))

    # setting all the needed information
    for st in all_subtrees:
        if pred_name == lemmatization(st):
            tag = re.search("\\'(.*)\\'", str(st.label()).split('\n')[0], re.IGNORECASE).group(1)
            node = {'pred': str(pred_name), 'tag': tag}

            # setting based on which feature is inside the subtree
            if 'NUM' in st.label().keys():
                node['num'] = st.label()['NUM']
            if 'TNS' in st.label().keys():
                node['tns'] = st.label()['TNS']
            if tag in ['NNS', 'NN'] and 'GEN' in st.label().keys():
                node['gen'] = st.label()['GEN']
            if 'LOC' in st.label().keys():
                node['loc'] = True
            return node
    return None


def lemmatization(term):
    """
    Given a term that represent a predicate in a tree, it returns the name of
    the predicate (lemma). For example, given "image(x,y)", it returns "image".

    :param term: the term (predicate) inside the tree, to find the lemma
    :return: the found predicate name (lemma) of the given term
    """
    tag = re.search("\\'(.*)\\'", str(term.label()).split('\n')[0], re.IGNORECASE).group(1)

    if tag == 'VBZ':
        term = term.label()['SEM'].term
        terms = subterms(term)
        terms = list(map(lambda x: x.pred.variable.name, terms))  # [<ConstantExpression presence>, <ConstantExpression agent>]
        if terms:
            return terms[0]  # presence
        else:
            return ""  # Needed because of lambda calculus, it avoids error when evaluating "is" from 2nd sentence
    if tag == 'VBG':
        term = term.label()['SEM'].term
        terms = subterms(term)
        # I need to split the return because of the different Lambda Calculus.
        if str(terms[0]).__contains__("image"):
            terms = list(map(lambda x: x.argument.term, terms))
            return terms[0].pred.variable.name  # image, 1st sentence
        else:
            return terms[0].pred.variable.name  # fly, 3rd sentence
    if tag == 'NN':
        term = term.label()['SEM'].term
        terms = subterms(term)
        terms = list(map(lambda x: x.pred.variable.name, terms))
        pred_name = terms[0]  # price, head
        return pred_name
    if tag == 'NNS':
        term = term.label()['SEM'].term
        terms = subterms(term)
        terms = list(map(lambda x: x.pred.variable.name, terms))
        return terms[0]  # thing
    if tag == 'PRP' and 'PERS' in term.label().keys():  # Ex. \P.P(you) -> you
        term = term.label()['SEM'].term
        return term.argument.variable.name
    elif tag == 'IN' and 'LOC' in term.label().keys():  # Ex. \R P x.R(\y.(P(x) & on(x,y))) -> on
        term = subterms(term.label()['SEM'].term)[0]
        return term.argument.term.second.pred.variable.name
    else:
        term = term.label()['SEM'].term
        terms = subterms(term)
    # For all other Variables and functions calculated by the Lambda Calculus (such as \P)
    terms = list(map(lambda x: x.pred.variable.name, terms))
    return terms[0] if len(terms) > 0 else None
