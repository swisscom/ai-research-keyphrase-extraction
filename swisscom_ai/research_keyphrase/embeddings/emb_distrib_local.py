import os
import pty
import subprocess

import numpy as np

from swisscom_ai.research_keyphrase.embeddings.emb_distrib_interface import EmbeddingDistributor


class EmbeddingDistributorLocal(EmbeddingDistributor):
    """
    Concrete class of @EmbeddingDistributor using a local installation of sent2vec
    https://github.com/epfml/sent2vec

    It works by creating a subprocess in which text are fed through stdin , and embeddings are read from the stdout
    This is temporary, we are waiting for the new version of the FastText Python Wrapper for a cleaner implementation.
    """

    def __init__(self, fasttext_path, fasttext_model):
        master, slave = pty.openpty()
        self._proc = subprocess.Popen(fasttext_path+' print-sentence-vectors '+fasttext_model, shell=True,
                                      stdin=subprocess.PIPE, stdout=slave, bufsize=1)
        self._stdin_handle = self._proc.stdin
        self._stdout_handle = os.fdopen(master)

    def get_tokenized_sents_embeddings(self, sents):
        """
        @see EmbeddingDistributor
        """
        for sent in sents:
            if '\n' in sent:
                raise RuntimeError('New line is not allowed inside a sentence')

        sentence_lines = '\n'.join(sents)+'\n'
        self._stdin_handle.write(sentence_lines.encode())
        self._stdin_handle.flush()
        all_embeddings = []
        for _ in sents:
            res = self._stdout_handle.readline()[:-2]  # remove last space and jumpline
            all_embeddings.append(eval('[' + res.replace(' ', ',') + ']'))
        return np.array(all_embeddings)
