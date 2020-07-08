from part2.exercise1.src.cs.conceptual_similarity import conceptual_similarity
from part2.exercise1.src.wsd.word_sense_disambiguation import word_sense_disambiguation

"""
In this file, we executes both Task 1 (conceptual similarity) and Task 2 (word 
sense disambiguation).
"""

if __name__ == "__main__":
    options_conceptual_similarity = {
        "input": "/Users/lorenzotabasso/Desktop/University/TLN/Progetto/19-20/tln-1920/part2/exercise1/input"
                 "/WordSim353.csv",
        "output": "/Users/lorenzotabasso/Desktop/University/TLN/Progetto/19-20/tln-1920/part2/exercise1/output/",
    }

    options_word_sense_disambiguation = {
        "input": "/Users/lorenzotabasso/Desktop/University/TLN/Progetto/19-20/tln-1920/part2/exercise1/input/semcor3"
                 ".0/brown1/tagfiles/br-a01",
        "output": "/Users/lorenzotabasso/Desktop/University/TLN/Progetto/19-20/tln-1920/part2/exercise1/output/"
    }

    print("Task 1: Conceptual Similarity")
    # conceptual_similarity(options_conceptual_similarity)

    print("\nTask 2: Word Sense Disambiguation")
    print("[2] - Running Lesks's algorithm...")
    word_sense_disambiguation(options_word_sense_disambiguation)
    print("[2] - Done.")
