
from textinator import Textinator

STOP_WORDS_FILENAME = 'stop_words.txt'

ASPECT_LIST = ['health effects of cigarettes',
                       'health impact',
                       'health effects of e-cigarettes',
                       'health effects of vapes',
                       'impact on lungs',
                       'impact on heart',
                       'cigarettes',
                       'e-cigarettes']

WORD_LIST = ["exposure", "risk", "mortality",
                                  "cardiovascular", "ecigarettes",
                                  "smoking", "asthma", "airway"]

def main():
    T = Textinator()

    T.load_stop_words(STOP_WORDS_FILENAME)

    T.load_text('data/cig_data/independent_1.pdf', 'I1', parser=T.pdf_parser, GPT = False)
    T.load_text('data/cig_data/independent_2.pdf', 'I2', parser=T.pdf_parser, GPT = False)
    T.load_text('data/cig_data/independent_3.pdf', 'I3', parser=T.pdf_parser, GPT = False)
    T.load_text('data/cig_data/independent_4.pdf', 'I4', parser=T.pdf_parser, GPT = False)
    T.load_text('data/cig_data/independent_5.pdf', 'I5', parser=T.pdf_parser, GPT = False)

    T.load_text('data/cig_data/industry_sponsored_1.pdf', 'S1',
                parser=T.pdf_parser, GPT = False)
    T.load_text('data/cig_data/industry_sponsored_2.pdf', 'S2',
                parser=T.pdf_parser, GPT = False)
    T.load_text('data/cig_data/industry_sponsored_3.pdf', 'S3',
                parser=T.pdf_parser, GPT = False)
    T.load_text('data/cig_data/industry_sponsored_4.pdf', 'S4',
                parser=T.pdf_parser, GPT = False)
    T.load_text('data/cig_data/industry_sponsored_6.txt', 'S5', GPT = False)

    T.ASBA_scores('data/GPT_sectioned/industry_sponsored_1.txt', ASPECT_LIST)
    T.ASBA_scores('data/GPT_sectioned/industry_sponsored_2.txt', ASPECT_LIST)
    T.ASBA_scores('data/GPT_sectioned/industry_sponsored_3.txt', ASPECT_LIST)
    T.ASBA_scores('data/GPT_sectioned/industry_sponsored_4.txt', ASPECT_LIST)
    T.ASBA_scores('data/GPT_sectioned/industry_sponsored_5.txt', ASPECT_LIST)
    T.ASBA_scores('data/GPT_sectioned/independent_1.txt', ASPECT_LIST)
    T.ASBA_scores('data/GPT_sectioned/independent_2.txt', ASPECT_LIST)
    T.ASBA_scores('data/GPT_sectioned/independent_3.txt', ASPECT_LIST)
    T.ASBA_scores('data/GPT_sectioned/independent_4.txt', ASPECT_LIST)
    T.ASBA_scores('data/GPT_sectioned/independent_5.txt', ASPECT_LIST)
    T.ASBA_scores('data/GPT_sectioned/independent_6.txt', ASPECT_LIST)

    T.wordcount_sankey(word_list=WORD_LIST)
    T.sentiment_analysis()
    T.LDA(5, 1, 5)
    df = T.dot_plot_df()
    T.create_dot_plot(df)

if __name__ == '__main__':
    main()