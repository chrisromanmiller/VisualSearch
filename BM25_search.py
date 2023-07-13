# Import BM25 library and other necessary libraries
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd
import os
import string




def cleaner(input_string):
    punctuations = string.punctuation + "-"
    translation_table = str.maketrans("", "", punctuations)
    processed_string = input_string.translate(translation_table)
    return processed_string.lower()




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
    
    # Take the query and compare it with every title and return indices with the highest scores
    doc_scores = bm25.get_scores(tokenized_query)
    
    return doc_scores




def bm25search_get_n_ind(query, df, n):

    doc_scores = bm25search_get_scores(query, df)
    indices = np.flip(np.argsort(doc_scores))
    
    return indices[0:n]




def bm25search_get_nonzero_ind(query, df):

    doc_scores = bm25search_get_scores(query, df)
    indices = np.flip(np.argsort(doc_scores))
   
    end_ind = len(indices) - 1
    
    for num, i in enumerate(indices):
        if doc_scores[i] == 0:
            end_ind = num
            break
            
    return indices[0:end_ind]
    
    
    
    
    

    
def bm25search_get_n_ids(query, df, n):

    ind = bm25search_get_n_ind(query, df, n)
    df_subset = df.iloc[ind]
    ids = df_subset.id.astype(str).tolist()
    
    return ids, df_subset




def bm25search_get_nonzero_ids(query, df):

    ind = bm25search_get_nonzero_ind(query, df)
    df_subset = df.iloc[ind]
    ids = df_subset.id.astype(str).tolist()
    
    return ids, df_subset





