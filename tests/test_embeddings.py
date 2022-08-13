from swisscom_ai.research_keyphrase.embeddings.emb_distrib_local import EmbeddingDistributorLocal


def test_basic_embedding():
    import sent2vec
    print(dir(sent2vec))
    emb = EmbeddingDistributorLocal("tests/resources/example_model")
    result = emb.get_tokenized_sents_embeddings(["Hello world"])

    assert result
