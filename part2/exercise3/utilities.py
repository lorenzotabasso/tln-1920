from nltk.corpus import stopwords
import nltk


def bag_of_word(text):
    """
    Support function, it returns the Bag of Word representation fo the given text.
    It applies lemmatization, removes the punctuation, the stop-words and duplicates.
    :param text: input text
    :return: Bag of Words representation of the text.
    """

    text = text.lower()
    stop_words = set(stopwords.words('english'))
    punct = {',', ';', '(', ')', '{', '}', ':', '?', '!', '‘'}
    wnl = nltk.WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tokens = list(filter(lambda x: x not in stop_words and x not in punct, tokens))
    return set(wnl.lemmatize(t) for t in tokens)


def create_vectors(topic, nasari):
    """
    It creates a list of Nasari vectors (a list of {term:score}). Every vector
    is linked to one topic term.
    :param topic: the list of topic's terms
    :param nasari: Nasari dictionary
    :return: list of Nasari's vectors.
    """

    vectors = []
    for word in topic:
        if word in nasari.keys():
            vectors.append(nasari[word])

    return vectors


# TODO: fondere create_vectors e create_context
def create_context(text, nasari):
    """
    It creates a list of Nasari vectors (a list of {term:score}). Every vector
    is linked to one text term.
    :param text: the list of text's terms
    :param nasari: Nasari dictionary
    :return: list of Nasari's vectors.
    """

    tokens = bag_of_word(text)
    vectors = []
    for word in tokens:
        if word in nasari.keys():
            vectors.append(nasari[word])

    return vectors


def get_title_topic(document, nasari):
    """
    Creates a list of Nasari vectors based on the document's title.
    :param document: input document
    :param nasari: Nasari dictionary
    :return: a list of Nasari vectors.
    """

    title = document[0]
    tokens = bag_of_word(title)
    vectors = create_vectors(tokens, nasari)
    return vectors
