from swisscom_ai.research_keyphrase.embeddings.emb_distrib_interface import (
    EmbeddingDistributor,
)
import numpy as np

from sentence_transformers import SentenceTransformer


class EmbeddingDistributorSBert(EmbeddingDistributor):
    def __init__(self, model: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model)

    def get_tokenized_sents_embeddings(self, sents):
        """
        @see EmbeddingDistributor
        """
        embeddings = []
        for sent in sents:
            embeddings.append(self.model.encode(sent))

        return np.asarray(embeddings)
