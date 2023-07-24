import pandas as pd
import numpy as np
import os
from scipy.special import softmax

from BM25_search import *
from clip_search import *
from paths import styles_path, images_path



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





    # BM25 text search scores
    if (query != "" and image is None and query2 == ""):
        bm_scores = bm25search_get_scores(query, df)
    elif (query != ""):
        bm_scores = np.zeros(length)
        for word in query.split(' '):
            bm_scores = bm25search_get_scores(word, df)
            bm_scores = np.add(bm_scores, np.not_equal(np.zeros_like(bm_scores), bm_scores))
    else:
        bm_scores = np.ones(length)



    # CLIP text search scores
    if (query2 != "" and image is None):
        distances = clip_text_search_get_distances(query2, df)
        clip_distances = distances[0]
        clip_text_scores = softmax(-clip_distances)
        clip_text_scores = clip_text_scores / np.max(clip_text_scores)
    elif (query2 == ""):
        clip_text_scores = np.ones(length)



    # CLIP image search scores
    if (image is not None and query2 == ""):
        distances = clip_image_search_get_distances(image, df)
        clip_distances = distances[0]
        clip_image_scores = softmax(-clip_distances)
        clip_image_scores = clip_image_scores / np.max(clip_image_scores)
    elif (image is None):
        clip_image_scores = np.ones(length)


    # CLIP combined image and text search scores
    if (image is not None and query2 != ""):
        D_image, D_text = clip_image_and_text_search_get_distances(image, query2, df)
        #D_image = clip_image_search_get_distances(image, df)
        #D_text = clip_text_search_get_distances(query2, df)

        clip_image_distances = D_image[0]
        clip_image_scores = softmax(-clip_image_distances)
        clip_image_scores = clip_image_scores / np.max(clip_image_scores)

        clip_text_distances = D_text[0]
        clip_text_distances = clip_text_distances / np.max(clip_text_distances)
        clip_text_scores_pre = softmax(-clip_text_distances)
        sorted = np.flip(np.sort(clip_text_scores_pre))
        clip_text_scores = np.zeros(length)
        for i in range(0, length, 100):
            clip_text_scores = np.add(np.less_equal(sorted[i]*np.ones(length), clip_text_scores_pre), clip_text_scores)




    # Combines results
    result = np.multiply(clip_image_scores, clip_text_scores)
    result = np.multiply(result, bm_scores)



    # Sorting results in descending order and returning an array of indices
    # corresponding to the sorted scores in results
    indices = np.flip(np.argsort(result)).tolist()

    # Takes the top 20 indices and locates the top 20 products in the dataframe
    df_subset = df.iloc[indices[0:20]]

    # Creates a list of image ids from the top 20 products
    ids = df_subset.id.astype(str).tolist()


    if (query == "" and query2 == "" and image is None):
        return [], df


    return ids, df
