import re
import sys
from optparse import OptionParser
import xml.etree.ElementTree as ET
from lxml import etree as Exml
from part2.exercise1.utilities import *

def parse_xml(path):
    """
    Parsifica SemCor Corpus annotato manualmente sui Synset di WordNet

    1) carico file xml
    2) prendo tutti tag s
    3) ricavo la frase
    4) selezione delle parole da disambiguare (selezionando quelle che servono) con num sensi >=2
    5) ricavo senso annotato (golden) da wsn

    :param path: percorso del file XML (Brown Corpus)
    :return: [(sentence, [(word, gold)])]
    """

    with open(path, 'r') as fileXML:
        data = fileXML.read()

        # correzioni xml non ben formattato
        data = data.replace('\n', '')
        replacer = re.compile("=([\w|:|\-|$|(|)|']*)")
        data = replacer.sub(r'="\1"', data)

        result = []
        try:
            root = Exml.XML(data)
            paragraphs = root.findall("./context/p")
            sentences = []
            for p in paragraphs:
                sentences.extend(p.findall("./s"))
            for sentence in sentences:
                words = sentence.findall('wf')
                sent = ""
                tuple_list = []
                for word in words:
                    w = word.text
                    pos = word.attrib['pos']
                    sent = sent + w + ' '
                    if pos == 'NN' and '_' not in w and len(wn.synsets(w)) > 1 and 'wnsn' in word.attrib:
                        sense = word.attrib['wnsn']
                        t = (w, sense)
                        tuple_list.append(t)
                result.append((sent, tuple_list))
        except Exception as e:
            raise NameError('xml: ' + str(e))
    return result


def exercise1():
    """
    Exercise 1: Extracts sentences from the SemCor corpus (corpus annotated
    with WN synset) and disambiguates at least one noun per sentence. It also
    calculates the accuracy based on the senses noted in SemCor. Writes the
    output into a xml file.
    """

    list_xml = parse_xml(options.input)

    result = []
    count_word = 0
    count_exact = 0

    for i in range(len(list_xml)):
        dict_list = []
        sentence = list_xml[i][0]
        words = list_xml[i][1]
        for t in words:
            sense = lesk(t[0], sentence)
            value = str(get_sense_index(t[0], sense))
            golden = t[1]
            count_word += 1
            if golden == value:
                count_exact += 1
            dict_list.append({'word': t[0], 'gold': golden, 'value': value})

        if len(dict_list) > 0:
            result.append((sentence, dict_list))

    accuracy = count_exact / count_word

    with open(options.output + 'exercise1_output.xml', 'wb') as out:
        out.write('<results accurancy="{0:.2f}">'.format(accuracy).encode())
        for i in range(len(result)):
            xml_s = ET.Element('sentence_wrapper')
            xml_s.set('sentence_number', str(i + 1))
            xml_sentence = ET.SubElement(xml_s, 'sentence')
            xml_sentence.text = result[i][0]
            for tword in result[i][1]:
                xml_word = ET.SubElement(xml_sentence, 'word')
                xml_word.text = tword['word']
                xml_word.set('golden', tword['gold'])
                xml_word.set('sense', tword['value'])

            tree = ET.ElementTree(xml_s)
            tree.write(out)

        out.write(b'</results>')


if __name__ == "__main__":
    print("Running Lesks's algorithm...")

    argv = sys.argv[1:]
    parser = OptionParser()

    parser.add_option("-b", "--input", help='path to the input file', action="store", type="string", dest="input",
                      default="/Users/lorenzotabasso/Desktop/University/TLN/Progetto/19-20/tln-1920/part2/exercise1/input/br-a01.xml")

    parser.add_option("-o", "--output", help='output directory', action="store", type="string", dest="output",
                      default="/Users/lorenzotabasso/Desktop/University/TLN/Progetto/19-20/tln-1920/part2/exercise1/output/")

    (options, args) = parser.parse_args()

    if options.input is None:
        print("Missing mandatory parameters")
        sys.exit(2)

    exercise1()

    # sentence = 'I arrived at the bank after crossing the river'
    # sentence = 'I arrived at the bank after crossing the road'
