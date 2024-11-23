"""
Author: <NAME>
Purpose: Load files, loads stopwords, does wordcount, sentiment analysis,
anything that we want to visualize in the visionator
Output:
"""
import default

from collections import defaultdict, Counter


class Textinator:
    def __init__(self):
        """ Constructor

        datakey --> (filelabel --> datavalue)
        """
        self.data = defaultdict(dict)
