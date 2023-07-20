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
import faiss


image_tensor_path = "./tensors/image_tensor.pt"
title_tensor_path = "./tensors/title_tensor.pt"

def clip_image_search_get_n_ids(path_to_image, df, n):

    custom_image = Image.open(path_to_image)

    # Loads model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    knn_vectors = torch.load(image_tensor_path)
    knn_labels = df.id.astype(str).tolist()

    # Convert data to numpy arrays for use with faiss
    vectors_np = knn_vectors.numpy()

    # Build the index
    dimension = vectors_np.shape[1]  # Dimension of the vectors
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors_np)

    # Preprocess the custom image
    custom_input = processor(text=[''], images=custom_image, return_tensors="pt", padding=True)

    # Perform inference on the custom image
    custom_output = model(**custom_input)

    # Get the custom image embedding
    custom_embedding = custom_output.image_embeds

    # Query the index with the custom image embedding
    k = n
    D, I = index.search(custom_embedding.detach().numpy(), k)

    # Retrieve the labels of the nearest neighbors if there are any
    neighbor_ids = [knn_labels[i] for i in I[0]]

    subset_df = df[df.id.isin(neighbor_ids)]

    return neighbor_ids, subset_df



def compute_distance_subset(index, xq, subset):
    n, _ = xq.shape
    _, k = subset.shape
    distances = np.empty((n, k), dtype=np.float32)
    index.compute_distance_subset(
        n, faiss.swig_ptr(xq),
        k, faiss.swig_ptr(distances),
        faiss.swig_ptr(subset)
    )
    return distances




def clip_image_search_get_distances(path_to_image, df):

    if type(path_to_image) is not np.ndarray:
        custom_image = Image.open(path_to_image)
    else:
        custom_image = path_to_image

    # Loads model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    knn_vectors = torch.load(image_tensor_path)
    knn_labels = df.id.astype(str).tolist()

    # Convert data to numpy arrays for use with faiss
    vectors_np = knn_vectors.numpy()

    # Build the index
    dimension = vectors_np.shape[1]  # Dimension of the vectors
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors_np)

    # Preprocess the custom image
    custom_input = processor(text=[''], images=custom_image, return_tensors="pt", padding=True)

    # Perform inference on the custom image
    custom_output = model(**custom_input)

    # Get the custom image embedding
    custom_embedding = custom_output.image_embeds

    # Query the index with the custom image embedding
    k = df.shape[0]
    distances, I = index.search(custom_embedding.detach().numpy(), k)
    distances[0] = distances[0][np.argsort(I[0])]

    return distances


def clip_text_search_get_distances(query, df):

    # Loads model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    knn_vectors = torch.load(title_tensor_path)
    knn_labels = df.id.astype(str).tolist()

    # Convert data to numpy arrays for use with faiss
    vectors_np = knn_vectors.numpy()

    # Build the index
    dimension = vectors_np.shape[1]  # Dimension of the vectors
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors_np)

    # Preprocess the custom image
    custom_input = processor(text=[''], images=custom_image, return_tensors="pt", padding=True)

    # Perform inference on the custom image
    custom_output = model(**custom_input)

    # Get the custom image embedding
    custom_embedding = custom_output.image_embeds

    # Query the index with the custom image embedding
    k = df.shape[0]
    distances, I = index.search(custom_embedding.detach().numpy(), k)
    distances[0] = distances[0][np.argsort(I[0])]

    return distances
