# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
#Authors: Kamil Bennani-Smires, Yann Savary

import argparse
import os
import re
import warnings
from abc import ABC, abstractmethod

# NLTK imports
import nltk
from nltk.tag.util import tuple2str
from nltk.parse import CoreNLPParser

import swisscom_ai.research_keyphrase.preprocessing.custom_stanford as custom_stanford
from swisscom_ai.research_keyphrase.util.fileIO import read_file, write_string

# If you want to use spacy , install it and uncomment the following import
# import spacy


class PosTagging(ABC):
    @abstractmethod
    def pos_tag_raw_text(self, text, as_tuple_list=True):
        """
        Tokenize and POS tag a string
        Sentence level is kept in the result :
        Either we have a list of list (for each sentence a list of tuple (word,tag))
        Or a separator [ENDSENT] if we are requesting a string by putting as_tuple_list = False

        Example :
        >>from sentkp.preprocessing import postagger as pt

        >>pt = postagger.PosTagger()

        >>pt.pos_tag_raw_text('Write your python code in a .py file. Thank you.')
        [
            [('Write', 'VB'), ('your', 'PRP$'), ('python', 'NN'),
            ('code', 'NN'), ('in', 'IN'), ('a', 'DT'), ('.', '.'), ('py', 'NN'), ('file', 'NN'), ('.', '.')
            ],
            [('Thank', 'VB'), ('you', 'PRP'), ('.', '.')]
        ]

        >>pt.pos_tag_raw_text('Write your python code in a .py file. Thank you.', as_tuple_list=False)

        'Write/VB your/PRP$ python/NN code/NN in/IN a/DT ./.[ENDSENT]py/NN file/NN ./.[ENDSENT]Thank/VB you/PRP ./.'


        >>pt = postagger.PosTagger(separator='_')
        >>pt.pos_tag_raw_text('Write your python code in a .py file. Thank you.', as_tuple_list=False)
        Write_VB your_PRP$ python_NN code_NN in_IN a_DT ._. py_NN file_NN ._.
        Thank_VB you_PRP ._.



        :param as_tuple_list: Return result as list of list (word,Pos_tag)
        :param text:  String to POS tag
        :return: POS Tagged string or Tuple list
        """

        pass

    def pos_tag_file(self, input_path, output_path=None):

        """
        POS Tag a file.
        Either we have a list of list (for each sentence a list of tuple (word,tag))
        Or a file with the POS tagged text

        Note : The jumpline is only for readibility purpose , when reading a tagged file we'll use again
        sent_tokenize to find the sentences boundaries.

        :param input_path: path of the source file
        :param output_path: If set write POS tagged text with separator (self.pos_tag_raw_text with as_tuple_list False)
                            If not set, return list of list of tuple (self.post_tag_raw_text with as_tuple_list = True)

        :return: resulting POS tagged text as a list of list of tuple or nothing if output path is set.
        """

        original_text = read_file(input_path)

        if output_path is not None:
            tagged_text = self.pos_tag_raw_text(original_text, as_tuple_list=False)
            # Write to the output the POS-Tagged text.
            write_string(tagged_text, output_path)
        else:
            return self.pos_tag_raw_text(original_text, as_tuple_list=True)

    def pos_tag_and_write_corpora(self, list_of_path, suffix):
        """
        POS tag a list of files
        It writes the resulting file in the same directory with the same name + suffix
        e.g
        pos_tag_and_write_corpora(['/Users/user1/text1', '/Users/user1/direct/text2'] , suffix = _POS)
        will create
        /Users/user1/text1_POS
        /Users/user1/direct/text2_POS

        :param list_of_path: list containing the path (as string) of each file to POS Tag
        :param suffix: suffix to append at the end of the original filename for the resulting pos_tagged file.

        """
        for path in list_of_path:
            output_file_path = path + suffix
            if os.path.isfile(path):
                self.pos_tag_file(path, output_file_path)
            else:
                warnings.warn('file ' + output_file_path + 'does not exists')


class PosTaggingStanford(PosTagging):
    """
    Concrete class of PosTagging using StanfordPOSTokenizer and StanfordPOSTagger

    tokenizer contains the default nltk tokenizer (PhunktSentenceTokenizer).
    tagger contains the StanfordPOSTagger object (which also trigger word tokenization  see : -tokenize option in Java).

    """

    def __init__(self, jar_path, model_path_directory, separator='|', lang='en'):
        """
        :param model_path_directory: path of the model directory
        :param jar_path: path of the jar for StanfordPOSTagger (override the configuration file)
        :param separator: Separator between a token and a tag in the resulting string (default : |)

        """

        if lang == 'en':
            model_path = os.path.join(model_path_directory, 'english-left3words-distsim.tagger')
            self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            self.tagger = custom_stanford.EnglishStanfordPOSTagger(model_path, jar_path, java_options='-mx2g')
        elif lang == 'de':
            model_path = os.path.join(model_path_directory, 'german-hgc.tagger')
            self.sent_tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')
            self.tagger = custom_stanford.GermanStanfordPOSTagger(model_path, jar_path, java_options='-mx2g')
        elif lang == 'fr':
            model_path = os.path.join(model_path_directory, 'french.tagger')
            self.sent_tokenizer = nltk.data.load('tokenizers/punkt/french.pickle')
            self.tagger = custom_stanford.FrenchStanfordPOSTagger(model_path, jar_path, java_options='-mx2g')
        else:
            raise ValueError('Language ' + lang + 'not handled')

        self.separator = separator

    def pos_tag_raw_text(self, text, as_tuple_list=True):
        """
        Implementation of abstract method from PosTagging
        @see PosTagging
        """
        tagged_text = self.tagger.tag_sents([self.sent_tokenizer.sentences_from_text(text)])

        if as_tuple_list:
            return tagged_text
        return '[ENDSENT]'.join(
            [' '.join([tuple2str(tagged_token, self.separator) for tagged_token in sent]) for sent in tagged_text])


class PosTaggingSpacy(PosTagging):
    """
        Concrete class of PosTagging using StanfordPOSTokenizer and StanfordPOSTagger
    """

    def __init__(self, nlp=None, separator='|' ,lang='en'):
        if not nlp:
            print('Loading Spacy model')
            #  self.nlp = spacy.load(lang, entity=False)
            print('Spacy model loaded ' + lang)
        else:
            self.nlp = nlp
        self.separator = separator

    def pos_tag_raw_text(self, text, as_tuple_list=True):
        """
            Implementation of abstract method from PosTagging
            @see PosTagging
        """

        # This step is not necessary int the stanford tokenizer.
        # This is used to avoid such tags :  ('      ', 'SP')
        text = re.sub('[ ]+', ' ', text).strip()  # Convert multiple whitespaces into one

        doc = self.nlp(text)
        if as_tuple_list:
            return [[(token.text, token.tag_) for token in sent] for sent in doc.sents]
        return '[ENDSENT]'.join(' '.join(self.separator.join([token.text, token.tag_]) for token in sent) for sent in doc.sents)
    

class PosTaggingCoreNLP(PosTagging):
    """
    Concrete class of PosTagging using a CoreNLP server 
    Provides a faster way to process several documents using since it doesn't require to load the model each time.
    """

    def __init__(self, host='localhost' ,port=9000, separator='|'):
        self.parser = CoreNLPParser(url=f'http://{host}:{port}')
        self.separator = separator
    
    def pos_tag_raw_text(self, text, as_tuple_list=True):
        # Unfortunately for the moment there is no method to do sentence split + pos tagging in nltk.parse.corenlp
        # Ony raw_tag_sents is available but assumes a list of str (so it assumes the sentence are already split)
        # We create a small custom function highly inspired from raw_tag_sents to do both

        def raw_tag_text():
            """
            Perform tokenizing sentence splitting and PosTagging and keep the 
            sentence splits structure
            """
            properties = {'annotators':'tokenize,ssplit,pos'}
            tagged_data = self.parser.api_call(text, properties=properties)
            for tagged_sentence in tagged_data['sentences']:
                yield [(token['word'], token['pos']) for token in tagged_sentence['tokens']]
        
        tagged_text = list(raw_tag_text())

        if as_tuple_list:
            return tagged_text
        return '[ENDSENT]'.join(
            [' '.join([tuple2str(tagged_token, self.separator) for tagged_token in sent]) for sent in tagged_text])
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write POS tagged files, the resulting file will be written'
                                                 ' at the same location with _POS append at the end of the filename')

    parser.add_argument('tagger', help='which pos tagger to use [stanford, spacy, corenlp]')
    parser.add_argument('listing_file_path', help='path to a text file '
                                                  'containing in each row a path to a file to POS tag')
    args = parser.parse_args()

    if args.tagger == 'stanford':
        pt = PosTaggingStanford()
        suffix = 'STANFORD'
    elif args.tagger == 'spacy':
        pt = PosTaggingSpacy()
        suffix = 'SPACY'
    elif args.tagger == 'corenlp':
        pt = PosTaggingCoreNLP()
        suffix = 'CoreNLP'

    list_of_path = read_file(args.listing_file_path).splitlines()
    print('POS Tagging and writing ', len(list_of_path), 'files')
    pt.pos_tag_and_write_corpora(list_of_path, suffix)
