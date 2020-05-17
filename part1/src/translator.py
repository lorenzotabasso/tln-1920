from nltk import parse
import sys


def main():
    my_parser = parse.load_parser("../grammars/my-simple-sem.fcfg")
    # sentence0 = "Angus gives a bone to every dog"
    sentence1 = "You are imagining things"
    sentence2 = "There is a price on my head"  # TODO: controllare grammatica
    # sentence3 = "Your big opportunity is flying out of here"
    # tokens = [sentence1.split(), sentence2.split(), sentence3.split()]
    tokens = [sentence1.split(), sentence2.split()]

    for sentences in tokens:
        for tree in my_parser.parse(sentences):
            print(tree.label()['SEM'])


if __name__ == "__main__":
    # argv = sys.argv[1:]
    main()
