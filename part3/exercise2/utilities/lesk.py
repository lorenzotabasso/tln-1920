import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords


def lesk(word, sentence):
    """
    Lesk's algoritm implementation. Given a word and a sentence in which it appears,
    it returns the best sense of the word.

    :param word: word to disabiguate
    :param sentence: sentence to compare
    :return: best sense of word
    """

    # Calculating the synset of the given word inside WN
    word_senses = wn.synsets(word)
    best_sense = word_senses[0]
    max_overlap = 0

    # I choose the bag of words approach
    context = bag_of_word(sentence)

    for sense in word_senses:
        # set of words in the gloss
        signature = bag_of_word(sense.definition())

        # and examples of the given sense
        examples = sense.examples()
        for ex in examples:
            # after this line, signature will contain for all the words, their
            # bag of words definition and their examples
            signature = signature.union(bag_of_word(ex))

        overlap = compute_overlap(signature, context)
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense

    return best_sense


def bag_of_word(sent):
    """
    Auxiliary function for the Lesk algorithm. Transforms the given sentence
    according to the bag of words approach, apply lemmatization, stop words
    and punctuation removal.

    :param sent: sentence
    :return: bag of words
    """

    stop_words = set(stopwords.words('english'))
    punct = {',', ';', '(', ')', '{', '}', ':', '?', '!'}
    wnl = nltk.WordNetLemmatizer()
    tokens = nltk.word_tokenize(sent)
    tokens = list(filter(lambda x: x not in stop_words and x not in punct, tokens))
    return set(wnl.lemmatize(t) for t in tokens)


def compute_overlap(signature, context):
    """
    Auxiliary function for the Lesk algorithm. Computes the number of words in
    common between signature and context.

    :param signature: bag of words of the signature (e.g. definitions + examples)
    :param context: bag of words of the context (e.g. sentence)
    :return: number of elements in commons
    """

    return len(signature & context)
