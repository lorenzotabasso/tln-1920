import nltk
import sys


def main():
    my_parser = nltk.parse.load_parser("../grammars/my-simple-sem.fcfg")

    sentence1 = "You are imagining things"
    sentence2 = "There is a price on my head"
    sentence3 = "Your big opportunity is flying out of here"
    tokens = [sentence1.split(), sentence2.split(), sentence3.split()]

    for sentences in tokens:
        for tree in my_parser.parse(sentences):
            print(tree.label()['SEM'])


if __name__ == "__main__":
    # argv = sys.argv[1:]
    main()
