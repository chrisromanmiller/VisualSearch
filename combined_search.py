import pandas as pd
import numpy as np
import os
from scipy.special import softmax

from BM25_search import *
from clip_search import *


styles_path = "./myntradataset/styles.csv"
images_path = "./myntradataset/images"


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
    if (query2 != ""):
        distances = clip_text_search_get_distances(query2, df)
        clip_distances = distances[0]
        clip_text_scores = softmax(-clip_distances)
        clip_text_scores = clip_text_scores / np.max(clip_text_scores)
    else:
        clip_text_scores = np.ones(length)



    # CLIP image search scores
    if (image is not None):
        distances = clip_image_search_get_distances(image, df)
        clip_distances = distances[0]
        clip_image_scores = softmax(-clip_distances)
        clip_image_scores = clip_image_scores / np.max(clip_image_scores)
    else:
        clip_image_scores = np.ones(length)





    result = np.multiply(bm_scores, clip_text_scores)
    result = np.multiply(result, clip_image_scores)


    indices = np.flip(np.argsort(result)).tolist()
    df_subset = df.iloc[indices[0:20]]
    ids = df_subset.id.astype(str).tolist()


    if (query == "" and query2 == "" and image is None):
        return [], df


    return ids, df
