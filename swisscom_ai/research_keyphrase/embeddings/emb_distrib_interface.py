# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
#Authors: Kamil Bennani-Smires, Yann Savary

from abc import ABC, abstractmethod


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class EmbeddingDistributor(ABC):
    """
    Abstract class in charge of providing the embeddings of piece of texts
    """
    @abstractmethod
    def get_tokenized_sents_embeddings(self, sents):
        """
        Generate a numpy ndarray with the embedding of each element of sent in each row
        :param sents: list of string (sentences/phrases)
        :return: ndarray with shape (len(sents), dimension of embeddings)
        """
        pass
