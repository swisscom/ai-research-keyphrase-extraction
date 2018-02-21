# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
#Authors: Kamil Bennani-Smires, Yann Savary

"""Implementation of StanfordPOSTagger with tokenization in the specific language, s.t. the tag and tag_sent methods
perform tokenization in the specific language.
"""
from nltk.tag import StanfordPOSTagger


class EnglishStanfordPOSTagger(StanfordPOSTagger):

    @property
    def _cmd(self):
        return ['edu.stanford.nlp.tagger.maxent.MaxentTagger',
                '-model', self._stanford_model, '-textFile', self._input_file_path,
                '-outputFormatOptions', 'keepEmptySentences']


class FrenchStanfordPOSTagger(StanfordPOSTagger):
    """
    Taken from github mhkuu/french-learner-corpus
    Extends the StanfordPosTagger with a custom command that calls the FrenchTokenizerFactory.
    """

    @property
    def _cmd(self):
        return ['edu.stanford.nlp.tagger.maxent.MaxentTagger',
                '-model', self._stanford_model, '-textFile',
                self._input_file_path, '-tokenizerFactory',
                'edu.stanford.nlp.international.french.process.FrenchTokenizer$FrenchTokenizerFactory',
                '-outputFormatOptions', 'keepEmptySentences']


class GermanStanfordPOSTagger(StanfordPOSTagger):
    """ Use english tokenizer for german """

    @property
    def _cmd(self):
        return ['edu.stanford.nlp.tagger.maxent.MaxentTagger',
                '-model', self._stanford_model, '-textFile', self._input_file_path,
                '-outputFormatOptions', 'keepEmptySentences']
