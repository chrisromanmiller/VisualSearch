# Import BM25 library and other necessary libraries
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd
import string






# This function eliminates punctuation characters and uncapitalizes letters from the input string

def cleaner(input_string):
    punctuations = string.punctuation + "-"
    translation_table = str.maketrans("", "", punctuations)
    processed_string = input_string.translate(translation_table)
    return processed_string.lower()






# This function computes all BM25 scores for each product stored in the pandas dataframe df given the string query
# Returns an array of scores in the same order as the dataframe df

def bm25search_get_scores(query, df):

    # Get titles from the dataframe
    titles = df.productDisplayName.tolist()
    cleaned_titles = [cleaner(doc) for doc in titles]
    tokenized_titles = [cleaned_title.split(" ") for cleaned_title in cleaned_titles]

    # Initialize BM25 API using image titles
    tokenized_titles
    bm25 = BM25Okapi(tokenized_titles)

    # Prepping query
    cleaned_query = cleaner(query)
    tokenized_query = cleaned_query.split(" ")

    # Take the query and compare it with every title and return an array of scores
    doc_scores = bm25.get_scores(tokenized_query)

    return doc_scores






# This function computes all BM25 scores for each product stored in the pandas dataframe df given the string query
# Returns an array of scores in descending order and returns an array of indices that indicate which scores belong to which product in the dataframe df

def bm25search_get_scores_and_indices(query, df):

    doc_scores = bm25search_get_scores(query, df)
    indices = np.flip(np.argsort(doc_scores))

    return doc_scores[indices], indices






# This function computes all BM25 scores for each product stored in the pandas dataframe df given the string query
# Returns an array of indices corresponding to the top n scores in descending order.

def bm25search_get_n_ind(query, df, n):

    doc_scores = bm25search_get_scores(query, df)
    indices = np.flip(np.argsort(doc_scores))

    return indices[0:n]






# This function computes all BM25 scores for each product stored in the pandas dataframe df given the string query
# Returns an array of indices corresponding to the BM25 scores in descending order all scores of 0 are omitted

def bm25search_get_nonzero_ind(query, df):

    doc_scores = bm25search_get_scores(query, df)
    indices = np.flip(np.argsort(doc_scores))

    end_ind = len(indices) - 1

    for num, i in enumerate(indices):
        if doc_scores[i] == 0:
            end_ind = num
            break

    return indices[0:end_ind]






# This function computes all BM25 scores for each product stored in the pandas dataframe df given the string query
# Returns an array of product ids corresponding to the top n scores in descending order. Also returns a subset of df containing the top n products

def bm25search_get_n_ids(query, df, n):

    ind = bm25search_get_n_ind(query, df, n)
    df_subset = df.iloc[ind]
    ids = df_subset.id.astype(str).tolist()

    return ids, df_subset






# This function computes all BM25 scores for each product stored in the pandas dataframe df given the string query
# Returns an array of ids corresponding to the BM25 scores in descending order all scores of 0 are omitted

def bm25search_get_nonzero_ids(query, df):

    ind = bm25search_get_nonzero_ind(query, df)
    df_subset = df.iloc[ind]
    ids = df_subset.id.astype(str).tolist()

    return ids, df_subset
