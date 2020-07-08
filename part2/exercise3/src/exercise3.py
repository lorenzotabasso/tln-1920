import sys
from pathlib import Path

from tqdm import tqdm

from part2.exercise3.src.utilities import get_title_topic, create_context


def compute_overlap(topic, paragraph):
    """
    Support function used in Weighted Overlap's function below.
    :param topic: Vector representation of the topic
    :param paragraph: Vector representation of the paragraph
    :return: intesection between the given parameters
    """

    return topic & paragraph


def rank(vector, nasari_vector):
    """
    Computes the rank of the given vector.
    :param vector: input vector
    :param nasari_vector: input Nasari vector
    :return: vector's rank (position inside the nasari_vector)
    """

    for i in range(len(nasari_vector)):
        if nasari_vector[i] == vector:
            return i + 1


def weighted_overlap(topic_nasari_vector, paragraph_nasari_vector):
    """
    Implementation of the Weighted Overlap metrics (Pilehvar et al.)
    :param topic_nasari_vector: Nasari vector representing the topic
    :param paragraph_nasari_vector: Nasari vector representing the paragraph
    :return: square-rooted Weighted Overlap if exist, 0 otherwise.
    """

    overlap_keys = compute_overlap(topic_nasari_vector.keys(),
                                   paragraph_nasari_vector.keys())

    overlaps = list(overlap_keys)

    if len(overlaps) > 0:
        # sum 1/(rank() + rank())
        den = sum(1 / (rank(q, list(topic_nasari_vector)) +
                       rank(q, list(paragraph_nasari_vector))) for q in overlaps)

        # sum 1/(2*i)
        num = sum(list(map(lambda x: 1 / (2 * x),
                           list(range(1, len(overlaps) + 1)))))

        return den / num

    return 0


def parse_nasari_dictionary():
    """
    It parse the Nasari input file, and it converts into a more convenient
    Python dictionary.
    :return: a dictionary representing the Nasari input file. Fomat: {word: {term:score}}
    """

    nasari_dict = {}
    with open(options["nasari"], 'r', encoding="utf8") as file:
        for line in file.readlines():
            splits = line.split(";")
            vector_dict = {}

            for term in splits[2:]:
                k = term.split("_")
                if len(k) > 1:
                    vector_dict[k[0]] = k[1]

            nasari_dict[splits[1].lower()] = vector_dict

    return nasari_dict


def summarization(document, nasari_dict, percentage):
    """
    Applies summarization to the given document, with the given percentage.
    :param document: the input document
    :param nasari_dict: Nasari dictionary
    :param percentage: reduction percentage
    :return: the summarization of the given document.
    """

    # getting the topics based on the document's title.
    topics = get_title_topic(document, nasari_dict)

    paragraphs = []
    i = 0
    # for each paragraph, except the title (document[0])
    for paragraph in document[1:]:
        context = create_context(paragraph, nasari_dict)

        paragraph_wo = 0  # Weighted Overlap average inside the paragraph.

        for word in context:
            # Computing WO for each word inside the paragraph.
            topic_wo = 0
            for vector in topics:
                topic_wo = topic_wo + weighted_overlap(word, vector)
            if topic_wo != 0:
                topic_wo = topic_wo / len(topics)

            # Sum all words WO in the paragraph's WO
            paragraph_wo += topic_wo

        if len(context) > 0:
            paragraph_wo = paragraph_wo / len(context)
            # append in paragraphs a tuple with the index of the paragraph (to
            # preserve order), he WO of the paragraph and the paragraph's text.
            paragraphs.append((i, paragraph_wo, paragraph))
        i += 1

    to_keep = len(paragraphs) - int(round((percentage / 100) * len(paragraphs), 0))

    # Sort by highest score and keeps all the important entries. From first to "to_keep"
    new_document = sorted(paragraphs, key=lambda x: x[1], reverse=True)[:to_keep]
    # Restore the original order.
    new_document = sorted(new_document, key=lambda x: x[0], reverse=True)
    # delete unnecessary fields (x[0] which contains the "i" and x[1] which
    # contains the WO of the paragraph) inside new_document in order to
    # keep oly the text (I associated to each paragraph a score based on the
    # "importance").
    new_document = list(map(lambda x: x[2], new_document))

    new_document = [document[0]] + new_document
    return new_document


def parse_document(file):
    """
    It parse the given document.
    :param file: input document
    :return: a list of all document's paragraph.
    """

    document = []
    data = file.read_text(encoding='utf-8')
    lines = data.split('\n')

    for line in lines:
        # If the "#" character is present, it means the line contains the
        # document original link. So, if the # is not present,
        # we have a normal paragraph to append to the list.
        if line != '' and '#' not in line:
            line = line[:-1]  # deletes the final "\n" character.
            document.append(line)

    return document


global options  # Dictionary containing all the script settings. Used everywhere.

if __name__ == "__main__":

    options = {
        "input": "input/text-documents",
        "nasari": "input/dd-small-nasari-15.txt",
        "output": "output/",
        "percentage": 10
    }

    print("Summarization.\nReduction percentage: {}".format(options["percentage"]))

    nasari_dict = parse_nasari_dictionary()

    # Inspecting the input files
    path = Path(options["input"])
    files_documents = list(path.glob('./*.txt'))

    # showing progress bar
    progress_bar = tqdm(desc="Percentage", total=4, file=sys.stdout)

    for file in files_documents:
        document = parse_document(file)

        # For each document, do some pretty print of the original.
        # It will be used later for comparison.
        with open(options["output"] + 'Original-' + file.name,
                  'w', encoding='utf-8') as out_original:
            for paragraph in document:
                out_original.write(paragraph + '\n')

        # For each document do summarization.
        sum_document = summarization(document, nasari_dict, options["percentage"])

        with open(options["output"] + str(options["percentage"]) + '-' + file.name,
                  'w', encoding='utf-8') as out_summarized:
            for paragraph in sum_document:
                out_summarized.write(paragraph + '\n')

        progress_bar.update(1)
