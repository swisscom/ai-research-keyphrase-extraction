import argparse
from configparser import ConfigParser

from swisscom_ai.research_keyphrase.embeddings.emb_distrib_local import EmbeddingDistributorLocal
from swisscom_ai.research_keyphrase.model.input_representation import InputTextObj
from swisscom_ai.research_keyphrase.model.method import MMRPhrase
from swisscom_ai.research_keyphrase.preprocessing.postagging import PosTaggingCoreNLP
from swisscom_ai.research_keyphrase.util.fileIO import read_file


def extract_keyphrases(embedding_distrib, ptagger, raw_text, N, lang, beta=0.55, alias_threshold=0.7):
    """
    Method that extract a set of keyphrases

    :param embedding_distrib: An Embedding Distributor object see @EmbeddingDistributor
    :param ptagger: A Pos Tagger object see @PosTagger
    :param raw_text: A string containing the raw text to extract
    :param N: The number of keyphrases to extract
    :param lang: The language
    :param beta: beta factor for MMR (tradeoff informativness/diversity)
    :param alias_threshold: threshold to group candidates as aliases
    :return: A tuple with 3 elements :
    1)list of the top-N candidates (or less if there are not enough candidates) (list of string)
    2)list of associated relevance scores (list of float)
    3)list containing for each keyphrase a list of alias (list of list of string)
    """
    tagged = ptagger.pos_tag_raw_text(raw_text)
    text_obj = InputTextObj(tagged, lang)
    return MMRPhrase(embedding_distrib, text_obj, N=N, beta=beta, alias_threshold=alias_threshold)


def load_local_embedding_distributor():
    config_parser = ConfigParser()
    config_parser.read('config.ini')
    sent2vec_model_path = config_parser.get('SENT2VEC', 'model_path')
    return EmbeddingDistributorLocal(sent2vec_model_path)


def load_local_corenlp_pos_tagger():
    config_parser = ConfigParser()
    config_parser.read('config.ini')
    host = config_parser.get('STANFORDCORENLPTAGGER', 'host')
    port = config_parser.get('STANFORDCORENLPTAGGER', 'port')
    return PosTaggingCoreNLP(host, port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract keyphrases from raw text')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-raw_text', help='raw text to process')
    group.add_argument('-text_file', help='file containing the raw text to process')
    

    parser.add_argument('-tagger_host', help='CoreNLP host', default='localhost')
    parser.add_argument('-tagger_port', help='CoreNLP port', default=9000)
    parser.add_argument('-N', help='number of keyphrases to extract', required=True, type=int)
    args = parser.parse_args()

    if args.text_file:
        raw_text = read_file(args.text_file)
    else:
        raw_text = args.raw_text

    embedding_distributor = load_local_embedding_distributor()
    pos_tagger = load_local_corenlp_pos_tagger(args.tagger_host, args.tagger_port)
    print(extract_keyphrases(embedding_distributor, pos_tagger, raw_text, args.N, 'en'))
