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
import pandas as pd
import vaderSentiment.vaderSentiment as vs
from sankey import show_sankey

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

    def filter_words(self, words):
        translation_table = str.maketrans(
            {"\n": "", "\t": "", "\r": "", "=": "",
             ",": "", "-": "", "(": "", ")": "",
             ".": "", ":": "", "?": ""})

        cleaned_words = []
        for i in words:
            i = i.translate(translation_table)
            i = i.lower()
            if i not in self.stop_list and i.isalpha() and len(i) > 2:
                cleaned_words.append(i)

        return cleaned_words

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
        cleaned_words = self.filter_words(words)

        wc = Counter(cleaned_words)
        num = len(cleaned_words)
        return {'wordcount': wc, 'numwords': num}


    def load_stop_words(self, stopwords_file):
        with open(stopwords_file) as infile:
            for i in infile:
                self.stop_list.append(i.strip())

    def wordcount_sankey(self, word_list = None, k = 5):
        # print(type(self.data))
        word_counts = pd.DataFrame()

        stacked_df = pd.DataFrame()

        if word_list is None:
            word_list = set()

            for text in self.data["wordcount"]:
                word_list = word_list.union(set(i[0] for i in self.data["wordcount"][text].most_common(k)))
            word_list = list(word_list)
            word_list.sort()
            for text in self.data["wordcount"]:
                word_counts["Words"] = word_list

            # for text in self.data["wordcount"].keys():
                word_counts["Text"] = text
                word_counts["Frequency"] = list(self.data["wordcount"][text][word] for word in word_list)
                stacked_df = pd.concat([stacked_df, word_counts], ignore_index=True, sort=False)
        else:
            for text in self.data["wordcount"]:
                word_counts["Words"] = word_list
                word_counts["Text"] = text
                word_counts["Frequency"] = list(self.data["wordcount"][text][word] for word in word_list)
                stacked_df = pd.concat([stacked_df, word_counts], ignore_index=True, sort=False)

        show_sankey(stacked_df, "Text", "Words", "Frequency")

    def sentiment_analysis(self):
        all_words = ""
        total_sentiment = []
        text_sentiments = []
        analyzer = vs.SentimentIntensityAnalyzer()

        for text in self.data["wordcount"]:
            for i in self.data["wordcount"][text].keys():
                total_sentiment.append(analyzer.polarity_scores(i)[
                                "compound"] * self.data["wordcount"][text][i])
            text_sentiments.append(sum(total_sentiment) / self.data["numwords"][text])
        print(text_sentiments)
            # print(self.data["wordcount"][text])


        # for i in self.data["wordcount"].keys():
        #     for word in self.data["wordcount"][i]:
        #         if word in word_list:
        # make_sankey()

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

    # print(T.data)
    # T.wordcount_sankey(k=5)
    T.sentiment_analysis()
if __name__ == '__main__':
    main()