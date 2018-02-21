# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
#Authors: Kamil Bennani-Smires, Yann Savary

"""Module containing helper function to process results of a solr query"""


def process_tagged_text(s):
    """
    Return a tagged_text as a list of sentence where each sentence is list of tuple (word,tag)
    :param s: string tagged_text coming from solr word1|tag1 word2|tag2[ENDSENT]word3|tag3 ...
    :return: (list of list of tuple) list of sentences where each sentence is a list of tuple (word,tag)
    """

    def str2tuple(tagged_token_text, sep='|'):
        loc = tagged_token_text.rfind(sep)
        if loc >= 0:
            return tagged_token_text[:loc], tagged_token_text[loc + len(sep):]
        else:
            raise RuntimeError('Problem when parsing tagged token '+tagged_token_text)

    result = []
    for sent in s.split('[ENDSENT]'):
        sent = [str2tuple(tagged_token) for tagged_token in sent.split(' ')]
        result.append(sent)
    return result
