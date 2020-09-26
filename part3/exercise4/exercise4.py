import nltk
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import collections
from utilities import create_vectors, weighted_overlap, aux_compute_overlap, sentences_cosine_similarity


def parse_nasari_dictionary():
    """
    It parse the Nasari input file, and it converts into a more convenient
    Python dictionary.

    :return: a dictionary representing the Nasari input file. Format: {word: {term:weight}}
    """

    nasari_dict = {}
    with open(config["nasari"], 'r', encoding="utf8") as file:
        for line in file:
            splits = line.split("\t")
            vector_dict = {}

            for term in splits[2:config["limit"]]:
                k = term.split("_")
                if len(k) > 1:
                    vector_dict[k[0]] = k[1]

            nasari_dict[splits[1].lower()] = vector_dict

    return nasari_dict


def tokenize_text(text):
    """
    It divides the text in groups. Each group is composed by w words.

    :param text: input text
    :return: list of sequences (list of all groups of word)
    """

    sequences = []
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    sentences=[]
    j = 0
    # TODO: sistemare l'append delle sentences
    for i in range(config["token_sequence_size"], len(tokens), config["token_sequence_size"]):
        sequences.append(tokens[j:i])
        sentences.append(' '.join( tokens[j:i]))
        j = i

    print("\tFound {} sequences".format(str(len(sequences))))
    return sequences, sentences


def segmentation():

    # input
    nasari = parse_nasari_dictionary()
    with open(config["input"]) as file:
        lines = file.readlines()
    text = ''.join(lines)

    # Text tokenization
    sequences, sentences = tokenize_text(text)

    # Compute similarity between neighbors
    similarities = list(np.zeros(len(sequences)))
    for i in range(1, len(sequences) - 1):
        prev = create_vectors(sequences[i - 1], nasari)
        current = create_vectors(sequences[i], nasari)
        next = create_vectors(sequences[i + 1], nasari)

        # Because the Weighted Overlap measures the similarity between two
        # vectors representing two sentences (couple term:weight), we need to
        # compute the Square Root Weighted Overlap, which produces as output
        # the similarity between the two sentences represented in vector notation.

        # Computing Square Root Weighted Overlap between prev and current
        # sentences
        similarity = []
        for x in prev:
            for w in current:
                similarity.append(math.sqrt(weighted_overlap(x, w)))
        left = max(similarity) if len(similarity) > 0 else 0

        # Computing Square Root Weighted Overlap between next and current
        # paragraphs
        similarity = []
        for x in next:
            for w in current:
                similarity.append(math.sqrt(weighted_overlap(x, w)))
        right = max(similarity) if len(similarity) > 0 else 0

        # Final similarity
        similarities[i] = (left + right) / 2

    del nasari
    
    # Compute clusters ---------------------------------------------------------
    sentences_similarities = np.array(similarities)
    data = sentences_similarities.reshape(-1, 1)

    #set the best cluster size within the minimum and maximum supplied Es 2,10
    clustersgroup_size_ranges = np.arange(2, 10) 
    clustersgroup_sizes = {}
    for size in clustersgroup_size_ranges:
        model = KMeans(n_clusters=size).fit(data)
        predictions = model.predict(data)
        clustersgroup_sizes[size] = silhouette_score(data, predictions)
    best_clustersgroup_size = max(clustersgroup_sizes, key=clustersgroup_sizes.get)
    print("The best cluster group size is:" +str( best_clustersgroup_size))
    
    # Compute Kmeans with best cluster groupsize 
    kmeans = KMeans(n_clusters=best_clustersgroup_size)
    kmeans.fit(data)
    matix_clusterized = kmeans.labels_
    print("The array of sencences referencing a cluster is:\n {}".format(matix_clusterized))
    
    # Print quantity allocated on each cluster
    quantity_per_cluster = collections.Counter(matix_clusterized)
    print("The number of elements allocated on each cluster is:\n\t {}".format(quantity_per_cluster))

    # Calculating beginning windows lenght based on sentences evenly splitted in contiguos clusters
    initial_window_size = len(matix_clusterized) / best_clustersgroup_size
    print("The initial window size is: {}".format(initial_window_size))
    
    # windows è la porzione di array che contiene l'id dei cluster per frasi
    windows_list = np.array_split(sentences, best_clustersgroup_size)
    print("Tmp sliced array: {}".format(windows_list))
    counter = 0
    prev_overlap = 0
    prev_elem=''
    
    final_list = [l.tolist() for l in windows_list]

    iterations_log=[]
    stable = False
    while not stable:
        stable = True
        print('START CYCLE')
        for i in range(len(final_list)):
         
              if i > 0:
                last_prev_vs_prev_similarity = sentences_cosine_similarity(
                    ' '.join(final_list[i-1][:-1]), final_list[i-1][-1])
                last_prev_vs_curr_similarity = sentences_cosine_similarity(
                    final_list[i-1][-1], ' '.join(final_list[i]))
                
                print("FORWARD:{} - OVERLAP - PREV {} - CURR {}".format(i, last_prev_vs_prev_similarity, last_prev_vs_curr_similarity))       
                # TODO: vedificare problemi di index nel primo cluster o nell'ultimo
                if last_prev_vs_curr_similarity > last_prev_vs_prev_similarity: # ORIGINALE, OK
                    stable = False
                    elem_tomove = final_list[i-1].pop(len(final_list[i-1]) - 1)
                    final_list[i].insert(0, elem_tomove)
                else:
                    first_curr_vs_curr_similarity = sentences_cosine_similarity(
                        ' '.join(final_list[i][1:]), final_list[i][0])
                    first_curr_vs_prev_similarity = sentences_cosine_similarity(
                        final_list[i][0], ' '.join(final_list[i - 1]))
                    print("BACKWARD:{} - OVERLAP - PREV {} - CURR {}".format(i,
                                                                             first_curr_vs_prev_similarity, first_curr_vs_curr_similarity))
                    if first_curr_vs_prev_similarity > first_curr_vs_curr_similarity:
                        stable = False
                        elem_tomove = final_list[i].pop(0)
                        final_list[i - 1].append(elem_tomove)
        #Just for logging                
        iteration_log_line=[]                
        for j in range(len(final_list)):
            if j==0:
                iteration_log_line.append(len(final_list[j]))
            else:
                iteration_log_line.append((len(final_list[j])+iteration_log_line[j-1]))
        iterations_log.append(iteration_log_line)
        print('END CYCLE')
    print(iterations_log)
    print('TADAAAA')
    print(final_list)

    # Plotting -----------------------------------------------------------------
    print("Plotting...")

    length = len(similarities)
    x = np.arange(0, length, 1)
    y = np.array(similarities)

    short = int(0.016 * length)
    long = int(0.08 * length)
    very_long = int(0.16 * length) # 19
    span = length / (config["segments_number"] + 1)

    # moving average
    data = pd.DataFrame(data=y)  # dataframe containing the similarity (the output on blue)
    # short_rolling = data.rolling(window=short).mean()
    # long_rolling = data.rolling(window=long).mean()
    ema_very_long = data.ewm(span=very_long, adjust=False).mean()
    ema_very_long.plot()

    # local minimum
    f = np.array(ema_very_long.to_numpy()).reshape((length,))
    inv_data_y = f * (-1)
    valley = signal.find_peaks_cwt(inv_data_y, np.arange(1, span))

    fig, ax = plt.subplots()
    ax.plot(x, y, label='blocks cohesion', color='c')
    plt.plot(x[valley], f[valley], "o", label="local minimum (" + str(span) + " span)", color='r')

    for x in separators:
        ax.axvline(x, color='k', linewidth=1)

    ax.set(xlabel='tokens sequence gap', ylabel='similarity',
           title='Block similarity')
    ax.grid()
    ax.legend(loc='best')

    # dd/mm/YY H:M:S
    now = datetime.now().strftime("Plot - %d.%m.%Y-%H:%M:%S")
    plt.savefig('./part3/exercise4/output/{}.png'.format(now))
    plt.show()
    print("Plot saved in output folder.")


global config  # Dictionary of the configuration. Used across all the script.

# separator values for our text.
separators = [19, 50, 85]

if __name__ == "__main__":
    config = {
        "input": "./part3/exercise4/input/snowden.txt",
        "output": "./part3/exercise4/output/",
        "nasari": "./part3/exercise4/resources/NASARI_lexical_english.txt",
        "limit": 14,  # first x elem of nasari vector
        "token_sequence_size": 25,
        "segments_number": 9
    }

    print('Starting segmentation...')
    segmentation()
    print('Segmentation ended.')
