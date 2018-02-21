# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
#Authors: Kamil Bennani-Smires, Yann Savary

from nltk.stem import PorterStemmer


class InputTextObj:
    """Represent the input text in which we want to extract keyphrases"""

    def __init__(self, pos_tagged, lang, stem=False, min_word_len=3):
        """
        :param pos_tagged: List of list : Text pos_tagged as a list of sentences
        where each sentence is a list of tuple (word, TAG).
        :param stem: If we want to apply stemming on the text.
        """
        self.min_word_len = min_word_len
        self.considered_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}
        self.pos_tagged = []
        self.filtered_pos_tagged = []
        self.isStemmed = stem
        self.lang = lang

        if stem:
            stemmer = PorterStemmer()
            self.pos_tagged = [[(stemmer.stem(t[0]), t[1]) for t in sent] for sent in pos_tagged]
        else:
            self.pos_tagged = [[(t[0].lower(), t[1]) for t in sent] for sent in pos_tagged]

        temp = []
        for sent in self.pos_tagged:
            s = []
            for elem in sent:
                if len(elem[0]) < min_word_len:
                    s.append((elem[0], 'LESS'))
                else:
                    s.append(elem)
            temp.append(s)

        self.pos_tagged = temp
        # Convert some language-specific tag (NC, NE to NN) or ADJA ->JJ see convert method.
        if lang in ['fr', 'de']:
            self.pos_tagged = [[(tagged_token[0], convert(tagged_token[1])) for tagged_token in sentence] for sentence
                               in
                               self.pos_tagged]
        self.filtered_pos_tagged = [[(t[0].lower(), t[1]) for t in sent if self.is_candidate(t)] for sent in
                                    self.pos_tagged]

    def is_candidate(self, tagged_token):
        """

        :param tagged_token: tuple (word, tag)
        :return: True if its a valid candidate word
        """
        return tagged_token[1] in self.considered_tags

    def extract_candidates(self):
        """
        :return: set of all candidates word
        """
        return {tagged_token[0].lower()
                for sentence in self.pos_tagged
                for tagged_token in sentence
                if self.is_candidate(tagged_token) and len(tagged_token[0]) >= self.min_word_len
                }


def convert(fr_or_de_tag):
    if fr_or_de_tag in {'NN', 'NNE', 'NE', 'N', 'NPP', 'NC', 'NOUN'}:
        return 'NN'
    elif fr_or_de_tag in {'ADJA', 'ADJ'}:
        return 'JJ'
    else:
        return fr_or_de_tag
