import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from transformers import CLIPTokenizer, FlaxCLIPTextModel
import faiss

from paths import image_tensor_path






# This function completes an image search using the CLIP Model. 
# The first argument is an image or a path to an image needs
# The pandas dataframe is also required to get the ids of the most relevant products
# n is the number of results we want to return

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






# This function computes distances from query image to every image in the dataset using the CLIP Model. 
# The first argument is an image or a path to an image needs
# The pandas dataframe is also required
# This function returns an array of distances and has the same order as df

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






# This function computes text and image distances from every product in the dataset using the CLIP Model. 
# The first argument is an image or a path to an image needs
# The second argument is the text query being used
# The pandas dataframe is also required
# This function returns two arrays of distances each having the same order as df

def clip_image_and_text_search_get_distances(path_to_image, query, df):

    if type(path_to_image) is not np.ndarray:
        custom_image = Image.open(path_to_image)
    else:
        custom_image = path_to_image

    # Loads model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    text = [query]

    inputs = processor(text=text, images=custom_image, return_tensors="pt", padding=True)
    outputs = model(**inputs)


    knn_vectors = torch.load(image_tensor_path)
    knn_labels = df.id.astype(str).tolist()
    vectors_np = knn_vectors.numpy()

    # Build the index
    dimension = query_np.shape[1]  # Dimension of the vectors
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors_np)

    # Text Search
    np_TP = outputs.text_embeds.detach().cpu().numpy()
    query_vector = np_TP.reshape(1, -1)
    k = df.shape[0]
    D_text, I_text = index.search(query_vector, k)
    D_text[0] = D_text[0][np.argsort(I_text[0])]

    # Image Search
    np_TP = outputs.image_embeds.detach().cpu().numpy()
    query_vector = np_TP.reshape(1, -1)
    k = df.shape[0]
    D_image, I_image = index.search(query_vector, k)
    D_image[0] = D_image[0][np.argsort(I_image[0])]

    return D_image, D_text

