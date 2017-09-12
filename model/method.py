from model.methods_embeddings import extract_candidates_embedding_for_doc, extract_doc_embedding, extract_sent_candidates_embedding_for_doc
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings


def _MMR(embdistrib, text_obj, candidates, X, beta, N, use_filtered):
    """
    Core method using Maximal Marginal Relevance in charge to return the top-N candidates

    :param embdistrib: embdistrib: embedding distributor see @EmbeddingDistributor
    :param text_obj: Input text representation see @InputTextObj
    :param candidates: list of candidates (string)
    :param X: numpy array with the embedding of each candidate in each row
    :param beta: hyperparameter beta for MMR (control tradeoff between informativeness and diversity)
    :param N: number of candidates to extract
    :param use_filtered: if true filter the text by keeping only candidate word before computing the doc embedding
    :return: list of the top-N candidates (or less if there are not enough candidates)
    """

    if len(candidates) >= N:  # If #candidates >= N result will be sorted

        doc_embedd = extract_doc_embedding(embdistrib, text_obj, use_filtered)  # Extract doc embedding
        doc_sim = cosine_similarity(X, doc_embedd.reshape(1, -1))

        doc_sim /= np.max(doc_sim)
        doc_sim = 0.5 + (doc_sim - np.average(doc_sim)) / np.std(doc_sim)

        sim_between = cosine_similarity(X)
        np.fill_diagonal(sim_between, np.NaN)

        sim_between /= np.nanmax(sim_between, axis=0)
        sim_between = 0.5 + (sim_between - np.nanmean(sim_between, axis=0)) / np.nanstd(sim_between, axis=0)

        selected_candidates = []
        unselected_candidates = [c for c in range(len(candidates))]

        j = np.argmax(doc_sim)
        selected_candidates.append(j)
        unselected_candidates.remove(j)

        for _ in range(N - 1):
            selec_array = np.array(selected_candidates)
            unselec_array = np.array(unselected_candidates)

            distance_to_doc = doc_sim[unselec_array, :]
            dist_between = sim_between[unselec_array][:, selec_array]
            if dist_between.ndim == 1:
                dist_between = dist_between[:, np.newaxis]
            j = np.argmax(beta * distance_to_doc - (1 - beta) * np.max(dist_between, axis=1).reshape(-1, 1))
            item_idx = unselected_candidates[j]
            selected_candidates.append(item_idx)
            unselected_candidates.remove(item_idx)
        return candidates[selected_candidates]
    else:
        # If candidates < N return all the candidates (not sorted)
        return candidates.tolist()


def MMRPhrase(embdistrib, text_obj, beta=0.5, N=10, use_filtered=True):
    """
    Extract N keyphrases

    :param embdistrib: embedding distributor see @EmbeddingDistributor
    :param text_obj: Input text representation see @InputTextObj
    :param beta: hyperparameter beta for MMR (control tradeoff between informativeness and diversity)
    :param N: number of keyphrases to extract
    :param use_filtered: if true filter the text by keeping only candidate word before computing the doc embedding
    :return: list of N keyphrases (or less if there are not enough candidates)
    """
    candidates, X = extract_candidates_embedding_for_doc(embdistrib, text_obj)

    if len(candidates) == 0:
        warnings.warn('No keyphrase extracted for this document')
        return []

    return _MMR(embdistrib, text_obj, candidates, X, beta, N, use_filtered)


def MMRSent(embdistrib, text_obj, beta=0.5, N=10, use_filtered=True):
    """

    Extract N key sentences

    :param embdistrib: embedding distributor see @EmbeddingDistributor
    :param text_obj: Input text representation see @InputTextObj
    :param beta: hyperparameter beta for MMR (control tradeoff between informativeness and diversity)
    :param N: number of key sentences to extract
    :param use_filtered: if true filter the text by keeping only candidate word before computing the doc embedding
    :return: list of N key sentences (or less if there are not enough candidates)
    """
    candidates, X = extract_sent_candidates_embedding_for_doc(embdistrib, text_obj)

    if len(candidates) == 0:
        warnings.warn('No keysentence extracted for this document')
        return []

    return _MMR(embdistrib, text_obj, candidates, X, beta, N, use_filtered)



