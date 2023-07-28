import pandas as pd
import numpy as np
import os
from scipy.special import softmax

from BM25_search import *
from clip_search import *
from paths import styles_path, images_path






# This function combines all modes of searching and is the function that combines scores into one overall score. 
# The first argument is an image or a path to an image needs but it can be None if there is no image being used in the search
# The second argument is the BM25 text search query, can be empty if BM25 is not used
# The third argument is the CLIP text search query, can be empty if CLIP text search is not used
# This function returns an array of 20 of the most relevant product ids from our dataset

def combined_search(image, query, query2):

    # Import Data
    filename = styles_path
    df = pd.read_csv(filename, on_bad_lines="skip") #some lines of the dataset fail due to excess commas

    # Data Wrangling
    available_ids = os.listdir(images_path)
    available_ids = [int(x.replace(".jpg","")) for x in available_ids]
    df = df[df.id.isin(available_ids)] #some images are not actually available
    df=df.dropna(subset='productDisplayName')

    length = df.shape[0]


    if (query == "" and image is None):
        return [], df
    if (query2 != "" and image is None):
        return [], df



    # BM25 text search scores
    if (query != "" and image is None and query2 == ""):
        bm_scores = bm25search_get_scores(query, df)
    else:
        bm_scores = np.ones(length)



    # CLIP text search scores
    if (query2 == ""):
        clip_text_scores = np.ones(length)



    # CLIP image search scores
    if (image is not None and query2 == ""):
        distances = clip_image_search_get_distances(image, df)
        clip_image_scores = distances[0]
    elif (image is None):
        clip_image_scores = np.ones(length)


    # CLIP combined image and text search scores
    if (image is not None and query2 != ""):
        D_image, D_text = clip_image_and_text_search_get_distances(image, query2, df)
        clip_image_scores = D_image[0]
        clip_text_scores = D_text[0]




    # Combines results
    result = np.add(clip_image_scores, 1.1*clip_text_scores)
    result = np.multiply(result, softmax(-bm_scores))



    # Sorting results in descending order and returning an array of indices
    # corresponding to the sorted scores in results
    indices = (np.argsort(result)).tolist()

    # Takes the top 20 indices and locates the top 20 products in the dataframe
    df_subset = df.iloc[indices[0:20]]

    # Creates a list of image ids from the top 20 products
    ids = df_subset.id.astype(str).tolist()


    return ids, df
