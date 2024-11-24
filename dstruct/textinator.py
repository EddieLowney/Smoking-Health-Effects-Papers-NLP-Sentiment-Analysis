"""
Author: <NAME>
Purpose: Load files, loads stopwords, does wordcount, sentiment analysis,
anything that we want to visualize in the visionator
Output:
"""
import random as rnd
from collections import defaultdict, Counter


class Textinator:
    def __init__(self):
        """ Constructor

        datakey --> (filelabel --> datavalue)
        """
        self.data = defaultdict(dict)

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

    def json_parser(filename):
        f = open(filename)
        raw = json.load(f)
        text = raw['text']
        words = text.split(" ")
        wc = Counter(words)
        num = len(words)

        return {'wordcount': wc, 'numwords': num}

        def pdf_parser(filename):
            f = open(filename)

        def load_stop_words(self, stopwords_file):
            pass
        
    