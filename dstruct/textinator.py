"""
Author: <NAME>
Purpose: Load files, loads stopwords, does wordcount, sentiment analysis,
anything that we want to visualize in the visionator
Output:
"""
import random as rnd
from collections import defaultdict, Counter
import pdfplumber
from pdfminer.high_level import extract_text
import json
import os

STOP_WORDS_FILENAME = 'data/stop_words.txt'

class Textinator:
    def __init__(self):
        """ Constructor
        datakey --> (filelabel --> datavalue)
        """
        self.data = defaultdict(dict)
        self.stop_list = list()

    def load_text(self, filename, label=None, parser=None):
        """ Register a document with the framework.
        Extract and store data to be used later by
        the visualizations """

        if parser is None:
            results = self.default_parser(filename)
        else:
            results = parser(filename)

        if label is None:
            label = filename

        for k, v in results.items():
            self.data[k][label] = v

    def default_parser(self, filename):
        """ Parse a standard text file and produce
        extract data results in the form of a dictionary. """

        results = {
            'wordcount': Counter("To be or not to be".split(" ")),
            'numwords': rnd.randrange(10, 50)
        }
        return results

    def json_parser(self, filename):
        f = open(filename)
        raw = json.load(f)
        text = raw['text']
        words = text.split(" ")
        wc = Counter(words)
        num = len(words)
        return {'wordcount': wc, 'numwords': num}

    def pdf_parser(self, filename):
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_name = f'data/converted_files/{base_name}.txt'
        text = extract_text(filename)
        with open(output_name, 'w') as file:
            file.write(text)

        words = text.split(" ")
        wc = Counter(words)
        num = len(words)
        return {'wordcount': wc, 'numwords': num}


    def load_stop_words(self, stopwords_file):
        with open(stopwords_file) as infile:
            for i in infile:
                self.stop_list.append(i.strip())

    def filter_words(self):
        # print(self.data['wordcount']["I1"])

        temp_copy = self.data["wordcount"]["I1"].copy()
        translation_table = str.maketrans({"\n": "", "\t": "", "\r": "", "=": "",
                                           ",": "", "-": "", "(": "", ")": "",
                                           ".": "", ":": "", "?": ""})
        for i in temp_copy:
            old_key = i
            i = i.translate(translation_table)
            if i.lower() in self.stop_list or i.isalpha() == False:
                del self.data["wordcount"]["I1"][i]
            else:
                self.data["wordcount"]["I1"][i] = self.data["wordcount"]["I1"][old_key]
                del self.data["wordcount"]["I1"][old_key]

        print(self.data["wordcount"]["I1"])
        # print(self.data["wordcount"]["I1"])
        # print((self.data["wordcount"]["I1"].keys()))



        # print(self.data)


        

def main():

    T = Textinator()
    T.load_stop_words(STOP_WORDS_FILENAME)

    T.load_text('data/cig_data/independent_1.pdf', 'I1', parser=T.pdf_parser)
    T.load_text('data/cig_data/independent_2.pdf', 'I1', parser=T.pdf_parser)
    T.load_text('data/cig_data/independent_3.pdf', 'I1', parser=T.pdf_parser)
    T.load_text('data/cig_data/independent_4.pdf', 'I1', parser=T.pdf_parser)
    T.load_text('data/cig_data/independent_5.pdf', 'I1', parser=T.pdf_parser)
    T.load_text('data/cig_data/independent_6.pdf', 'I1', parser=T.pdf_parser)
    T.load_text('data/cig_data/industry_sponsored_1.pdf', 'S1', parser=T.pdf_parser)
    T.load_text('data/cig_data/industry_sponsored_2.pdf', 'S2', parser=T.pdf_parser)
    T.load_text('data/cig_data/industry_sponsored_3.pdf', 'S3', parser=T.pdf_parser)
    T.load_text('data/cig_data/industry_sponsored_4.pdf', 'S4', parser=T.pdf_parser)
    T.load_text('data/cig_data/industry_sponsored_5.pdf', 'S5', parser=T.pdf_parser)
    T.load_text('data/cig_data/industry_sponsored_6.pdf', 'S6')

    T.filter_words()

if __name__ == '__main__':
    main()