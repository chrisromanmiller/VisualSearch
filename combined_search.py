import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from transformers import CLIPModel
from transformers import CLIPProcessor
import time

from BM25_search import *
from displaying_images import *
from clip_search import *



def combined_search(image, query):

    # Import Data
    filename = "./myntradataset/styles.csv"
    df = pd.read_csv(filename, on_bad_lines="skip") #some lines of the dataset fail due to excess commas

    # Data Wrangling
    available_ids = os.listdir("./myntradataset/images")
    available_ids = [int(x.replace(".jpg","")) for x in available_ids]
    df = df[df.id.isin(available_ids)] #some images are not actually available
    df=df.dropna(subset='productDisplayName')
    
    
    if (query is not None and image is None):
        ids, subset = bm25search_get_n_ids(query, df, 20)
        return ids, df
        
    
    if (query is None and image is not None):
        ids = clip_search_get_n_ids(image, df, 20)
        return ids, df
        
        
    if (query is not None and image is not None):
        ids1, subset_df1 = bm25search_get_nonzero_ids(query, df)
        ids2, subset_df2 = clip_search_get_n_ids(image, df, 100)
        new_ids = np.intersect1d(np.array(ids1), np.array(ids2)).tolist()
        
        return new_ids, df
    
    
        
        






