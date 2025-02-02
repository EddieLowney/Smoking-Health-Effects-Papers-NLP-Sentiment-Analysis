"""
Author: <NAME>
Purpose: Load files, loads stopwords, does wordcount, sentiment analysis,
anything that we want to visualize in the visionator
Output:
"""


from collections import defaultdict, Counter
from pdfminer.high_level import extract_text
import json
import os
import re
from myopenai import MyOpenAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import dotenv
import pandas as pd
import vaderSentiment.vaderSentiment as vs
from SANKEY import show_sankey


dotenv.load_dotenv()
api = MyOpenAPI()

STOP_WORDS_FILENAME = 'data/stop_words.txt'

class Textinator:
    def __init__(self):
        """ Constructor
        datakey --> (filelabel --> datavalue)
        """
        self.data = defaultdict(dict)
        self.stop_list = list()

    def GPT_key_sections(self, text, filename):
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_name = f'data/GPT_sectioned/{base_name}.txt'
        prompt = """Find the sections of this text that really contribute to\
        its meaning. Example: find the abstract, key defining sentences,\
        discussion, and conclusion of a research paper. Only return exactly\
        what the article says. TEXT: """
        prompt += text
        response_text = api.ask(prompt=prompt)
        response_text = re.sub(r"\*\*.*?\*\*", '', response_text)
        response_text = re.sub(r"-", '', response_text)
        with open(output_name, 'w') as file:
            file.write(response_text)

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
        """Given a list of words, removes all specified characters from each
        word and removes any words that, following the filtering, are not
        exclusively letters (isalpha), are in the stop words list, or are
        less than 3 characters"""

        # Creation of translation table to remove characters
        translation_table = str.maketrans(
            {"\n": "", "\t": "", "\r": "", "=": "",
             ",": "", "-": "", "(": "", ")": "",
             ".": "", ":": "", "?": "", ";": "", "[": "", "]": "", " ": ""})

        cleaned_words = []
        # Loop iterating through words, applying translate() and lowercase()
        for i in words:
            i = i.translate(translation_table)
            i = i.lower()
            # Conditional statement checking filtering conditions
            if i not in self.stop_list and i.isalpha() and len(i) > 2:
                cleaned_words.append(i)

        return cleaned_words

    def default_parser(self, filename):
        """ Parse a standard text file and produce
        extract data results in the form of a dictionary. """

        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_name = f'data/converted_files/{base_name}.txt'

        with open(filename, 'r') as file:
            text = file.read()
        self.GPT_key_sections(text, filename)

        with open(output_name, 'w') as file:
            file.write(text)

        # Filters words
        cleaned_words = self.filter_words(text.split(" "))
        # Gets word counts in a Counter datatype
        wc = Counter(cleaned_words)
        num = len(cleaned_words)

        return {'wordcount': wc, 'numwords': num}

    def json_parser(self, filename):
        f = open(filename)
        raw = json.load(f)
        text = raw['text']
        words = text.split(" ")
        wc = Counter(words)
        num = len(words)
        return {'wordcount': wc, 'numwords': num}


    def pdf_parser(self, filename):
        """Called to parse a PDF file. Extracts text. Uses a PDF library
        to extract text, calls filter_word() to clean the output, and outputs
        the cleaned words in the dictionary counter datatype. Also writes the
        most important parts of the text to separate files using GPT API"""
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_name = f'data/converted_files/{base_name}.txt'

        text = extract_text(filename)
        #Writes most important portions of text to separate files using GPT
        self.GPT_key_sections(text, filename)
        with open(output_name, 'w') as file:
            file.write(text)

        # Filters words
        cleaned_words = self.filter_words(text.split(" "))
        # Gets word counts in a Counter datatype
        wc = Counter(cleaned_words)
        num = len(cleaned_words)
        
        return {'wordcount': wc, 'numwords': num}

    def load_stop_words(self, stopwords_file):
        with open(stopwords_file) as infile:
            for i in infile:
                self.stop_list.append(i.strip())

    def ASBA_scores(self, filename):
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_name = f'results/ASBA/{base_name}.csv'
        with open(filename, "r") as file:
            text = file.read()

        # Load the ABSA model and tokenizer
        model_name = "yangheng/deberta-v3-base-absa-v1.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        classifier = pipeline("text-classification", model=model,
                              tokenizer=tokenizer, device = 'mps')

        result = {}
        for aspect in ['health effects of cigarettes',
                       'health impact',
                       'health effects of e-cigarettes',
                       'health effects of vapes',
                       'impact on lungs',
                       'impact on heart',
                       'cigarettes',
                       'e-cigarettes',
                       'rainbows and unicorns']:
            result[aspect] = classifier(text, text_pair=aspect)[0]
        df = pd.DataFrame.from_dict(result, orient='index')
        df.to_csv(output_name)
        return df

    def wordcount_sankey(self, word_list = None, k = 5):
        """"""
        word_counts = pd.DataFrame()
        stacked_df = pd.DataFrame()

        if word_list is None:
            word_list = set()

            for text in self.data["wordcount"]:
                word_list = word_list.union(
                set(i[0] for i in self.data["wordcount"][text].most_common(k)))
            word_list = list(word_list)
            word_list.sort()
            for text in self.data["wordcount"]:
                word_counts["Words"] = word_list

            # for text in self.data["wordcount"].keys():
                word_counts["Text"] = text
                word_counts["Frequency"] = list(self.data[
                                "wordcount"][text][word] for word in word_list)
                stacked_df = pd.concat([
                    stacked_df, word_counts], ignore_index=True, sort=False)
        else:
            for text in self.data["wordcount"]:
                word_counts["Words"] = word_list
                word_counts["Text"] = text
                word_counts["Frequency"] = list(self.data[
                                "wordcount"][text][word] for word in word_list)
                stacked_df = pd.concat([
                    stacked_df, word_counts], ignore_index=True, sort=False)
        show_sankey(stacked_df, "Text", "Words", vals = "Frequency")

    def sentiment_analysis(self):
        all_words = ""
        total_sentiment = []
        text_sentiments = []
        analyzer = vs.SentimentIntensityAnalyzer()

        for text in self.data["wordcount"]:
            for i in self.data["wordcount"][text].keys():
                total_sentiment.append(analyzer.polarity_scores(i)[
                                "compound"] * self.data["wordcount"][text][i])
            text_sentiments.append(sum(total_sentiment) / self.data[
                                                            "numwords"][text])

    def LDA(self):
        pass

def main():

    T = Textinator()
    T.load_stop_words(STOP_WORDS_FILENAME)

    # T.load_text('data/cig_data/independent_1.pdf', 'I1', parser=T.pdf_parser)
    # T.load_text('data/cig_data/independent_2.pdf', 'I2', parser=T.pdf_parser)
    # T.load_text('data/cig_data/independent_3.pdf', 'I3', parser=T.pdf_parser)
    # T.load_text('data/cig_data/independent_4.pdf', 'I4', parser=T.pdf_parser)
    # T.load_text('data/cig_data/independent_5.pdf', 'I5', parser=T.pdf_parser)
    # T.load_text('data/cig_data/independent_6.pdf', 'I6', parser=T.pdf_parser)
    # T.load_text('data/cig_data/industry_sponsored_1.pdf', 'S1', parser=T.pdf_parser)
    # T.load_text('data/cig_data/industry_sponsored_2.pdf', 'S2', parser=T.pdf_parser)
    # T.load_text('data/cig_data/industry_sponsored_3.pdf', 'S3', parser=T.pdf_parser)
    # T.load_text('data/cig_data/industry_sponsored_4.pdf', 'S4', parser=T.pdf_parser)
    # T.load_text('data/cig_data/industry_sponsored_5.pdf', 'S5', parser=T.pdf_parser)
    # T.load_text('data/cig_data/industry_sponsored_6.txt', 'S6')
    # T.ASBA_scores('data/GPT_sectioned/industry_sponsored_1.txt')
    # T.ASBA_scores('data/GPT_sectioned/industry_sponsored_2.txt')
    # T.ASBA_scores('data/GPT_sectioned/industry_sponsored_3.txt')
    # T.ASBA_scores('data/GPT_sectioned/industry_sponsored_4.txt')
    # T.ASBA_scores('data/GPT_sectioned/industry_sponsored_5.txt')
    # T.ASBA_scores('data/GPT_sectioned/industry_sponsored_6.txt')
    # T.ASBA_scores('data/GPT_sectioned/independent_1.txt')
    # T.ASBA_scores('data/GPT_sectioned/independent_2.txt')
    # T.ASBA_scores('data/GPT_sectioned/independent_3.txt')
    # T.ASBA_scores('data/GPT_sectioned/independent_4.txt')
    # T.ASBA_scores('data/GPT_sectioned/independent_5.txt')
    # T.ASBA_scores('data/GPT_sectioned/independent_6.txt')

    T.wordcount_sankey(k=5)
    T.sentiment_analysis()
    


    # print(T.data)
if __name__ == '__main__':
    main()