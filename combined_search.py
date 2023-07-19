import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.special import softmax

import torch
from torchvision import models, transforms
from PIL import Image
from transformers import CLIPModel
from transformers import CLIPProcessor
import time

from BM25_search import *
from displaying_images import *
from clip_search import *


styles_path = "./myntradataset/styles.csv"
images_path = "./myntradataset/images"


def combined_search(image, query):

    # Import Data
    filename = styles_path
    df = pd.read_csv(filename, on_bad_lines="skip") #some lines of the dataset fail due to excess commas

    # Data Wrangling
    available_ids = os.listdir(images_path)
    available_ids = [int(x.replace(".jpg","")) for x in available_ids]
    df = df[df.id.isin(available_ids)] #some images are not actually available
    df=df.dropna(subset='productDisplayName')

    length = df.shape[0]


    if (query != "" and image is None):
        bm_sm = bm25search_get_scores(query, df)
    elif (query != ""):
        bm_sm = np.zeros(length)
        for word in query.split(' '):
            bm_scores = bm25search_get_scores(word, df)
            bm_sm = np.add(bm_sm, np.not_equal(np.zeros_like(bm_scores), bm_scores))
    else:
        bm_sm = np.ones(length)



    if (image is not None):
        distances = clip_search_get_distances(image, df)
        clip_distances = distances[0]
        clip_sm = softmax(-clip_distances)
        clip_sm = clip_sm / np.max(clip_sm)
    else:
        clip_sm = np.ones(length)





    result = np.multiply(bm_sm, clip_sm)

    indices = np.flip(np.argsort(result)).tolist()
    df_subset = df.iloc[indices[0:20]]
    ids = df_subset.id.astype(str).tolist()


    if (query == "" and image is None):
        return [], df


    return ids, df
