from langdetect import detect_langs
from preprocessing.postagging import PosTaggingStanford
from model.input_representation import InputTextObj
from model.method import MMRPhrase
from configparser import ConfigParser
from embeddings.emb_distrib_local import EmbeddingDistributorLocal

def tagged_text_and_lang(jar_path, model_directory_path, text):
    """

    :param jar_path: stanford pos tagger jar path
    :param model_directory_path: directory containing all stanford pos models
    :param text: raw text (string) to pos tag
    :return: Tuple containing 1) List of list : Text pos_tagged as a list of sentences
        where each sentence is a list of tuple (word, TAG).
        2) String : detected language ('en' 'fr' or 'de')
    """

    lang_list = detect_langs(text)
    lang = getLang(lang_list)

    print('language : ', lang)

    if lang in ['en', 'fr', 'de']:
        ptagger = PosTaggingStanford(jar_path, model_directory_path, lang=lang)
    else:
        raise ValueError(lang + ' detected and not handled at the moment')

    list_tagged = ptagger.pos_tag_raw_text(text)

    #Convert some language-specific tag (NC, NE to NN) or ADJA ->JJ see convert method.
    if lang in ['fr', 'de']:
        list_tagged = [[(tagged_token[0], convert(tagged_token[1])) for tagged_token in sentence] for sentence in
                       list_tagged]
    return list_tagged, lang


def getLang(lang_list):
    for item in lang_list:
        if item.lang == 'en' or item.lang == 'de' or item.lang == 'fr':
            return item.lang
    return lang_list[0].lang


def convert(fr_or_de_tag):
    if fr_or_de_tag in {'NN', 'NNE', 'NE', 'N', 'NPP', 'NC', 'NOUN'}:
        return 'NN'
    elif fr_or_de_tag in {'ADJA', 'ADJ'}:
        return 'JJ'
    else:
        return fr_or_de_tag


def extract_keyphrases(embedding_distrib, jar_path, model_directory_path, raw_text, N):
    tagged, lang = tagged_text_and_lang(jar_path, model_directory_path, raw_text)
    text_obj = InputTextObj(tagged, lang)
    return MMRPhrase(embedding_distrib, text_obj, N=N)

if __name__ == '__main__':
    config = ConfigParser()
    config.read('config.ini')
    stanford_jar_path = config.get('STANFORDTAGGER', 'jar_path')
    model_directory_path = config.get('STANFORDTAGGER', 'model_directory_path')
    sent2vec_bin = config.get('SENT2VEC', 'bin_path')
    sent2vec_model = config.get('SENT2VEC', 'model_path')

    embedding_distributor = EmbeddingDistributorLocal(sent2vec_bin, sent2vec_model)

    raw_text = 'this is a test blabla end of file blabla. then we do this hello hello. finally hi however i like kamil'
    print(extract_keyphrases(embedding_distributor, stanford_jar_path, model_directory_path, raw_text, 2))
