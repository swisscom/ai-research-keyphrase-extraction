from swisscom_ai.research_keyphrase.embeddings.emb_distrib_interface import (
    EmbeddingDistributor,
)
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np


class EmbeddingDistributorDoc2Vec(EmbeddingDistributor):
    def __init__(self, model):
        self.model = Doc2Vec.load(model)

    def get_tokenized_sents_embeddings(self, sents):
        """
        @see EmbeddingDistributor
        """
        embeddings = []
        for sent in sents:
            emb = self.model.infer_vector([sent], alpha=0.01, steps=1000)
            embeddings.append(emb)

        return np.array(embeddings)
