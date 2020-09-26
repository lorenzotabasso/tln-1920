import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 


def weighted_overlap(topic_nasari_vector, paragraph_nasari_vector):
    """
    Implementation of the Weighted Overlap metrics.
    (https://www.aclweb.org/anthology/N15-1059.pdf)
    (https://image1.slideserve.com/3157875/weighted-overlap-pilehvar-et-al-acl-2013-l.jpg)

    :param topic_nasari_vector: Nasari vector representing the topic
    :param paragraph_nasari_vector: Nasari vector representing the paragraph
    :return: square-rooted Weighted Overlap if exist, 0 otherwise.
    """

    overlap_keys = aux_compute_overlap(topic_nasari_vector.keys(),
                                       paragraph_nasari_vector.keys())

    overlaps = list(overlap_keys)

    if len(overlaps) > 0:
        # sum 1/(rank() + rank())
        den = sum(1 / (aux_rank(q, list(topic_nasari_vector)) +
                       aux_rank(q, list(paragraph_nasari_vector))) for q in overlaps)

        # sum 1/(2*i)
        num = sum(list(map(lambda x: 1 / (2 * x),
                           list(range(1, len(overlaps) + 1)))))

        return den / num

    return 0


def aux_compute_overlap(bow_text_signature, bow_text_context):
    """
    Computes the number of words in common between signature and context.

    :param bow_text_signature: bag of words of the text's signature (e.g. definitions +
    examples)
    :param bow_text_context: bag of words of the context (e.g. sentence)
    :return: intersection between signature and context
    """

    return bow_text_signature & bow_text_context


def aux_rank(input_vector, nasari_vector):
    """
    Computes rank between the vector and the Nasari vector

    :param input_vector: input vector
    :param nasari_vector: Nasari vector
    :return: Rank of the input vector (position)
    """

    for i in range(len(nasari_vector)):
        if nasari_vector[i] == input_vector:
            return i + 1


def create_vectors(tokens, nasari):
    """
    It creates a list of Lexical Nasari vectors (a list of {term:weight}).
    Every vector is linked to one token of the text.

    :param tokens: input text
    :param nasari: Nasari dictionary
    :return: list of Nasari's vectors.
    """

    tokens = aux_bag_of_word(tokens)
    vectors = []
    for word in tokens:
        if word in nasari.keys():
            vectors.append(nasari[word])

    return vectors


def aux_bag_of_word(tokens):
    """
    It returns the Bag of Word representation fo the given text.
    It applies lemmatization, removes the punctuation, the stop-words and duplicates.

    :param tokens: input text
    :return: Bag of Words representation of the text.
    """

    stop_words = set(stopwords.words('english'))
    punct = {',', ';', '(', ')', '{', '}', ':', '?', '!', '*'}
    wnl = nltk.WordNetLemmatizer()
    tokens = list(filter(lambda x: x not in stop_words and x not in punct, tokens))
    return set(wnl.lemmatize(t) for t in tokens)


def sentences_cosine_similarity(x, y):
    # https://www.geeksforgeeks.org/python-measure-similarity-between-two-sentences-using-cosine-similarity/
    # Program to measure the similarity between  
    # two sentences using cosine similarity. 

    # tokenization 
    x_list = word_tokenize(x.lower())  
    y_list = word_tokenize(y.lower()) 
    
    # sw contains the list of stopwords 
    sw = stopwords.words('english')  
    l1 =[];l2 =[] 
    
    # remove stop words from the string 
    x_set = {w for w in x_list if not w in sw}  
    y_set = {w for w in y_list if not w in sw} 
    
    # form a set containing keywords of both strings  
    rvector = x_set.union(y_set)  
    for w in rvector: 
        if w in x_set: l1.append(1) # create a vector 
        else: l1.append(0) 
        if w in y_set: l2.append(1) 
        else: l2.append(0) 
    c = 0
    
    # cosine formula  
    for i in range(len(rvector)): 
            c+= l1[i]*l2[i] 
    cosine = c / float((sum(l1)*sum(l2))**0.5) 
    # print("similarity: ", cosine)
    return cosine
