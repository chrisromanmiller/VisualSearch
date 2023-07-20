import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import torch
import faiss
import os
import pandas as pd
from transformers import CLIPModel, CLIPProcessor
import sys

st.title("Mindblowing Demo")

def select_rectangle_with_point(image):
    # Make a copy of the image for display
    clone = image.copy()

    # Initialize variables for storing coordinates and selection status
    start_point = None
    end_point = None
    extra_point = None
    selection_completed = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal start_point, end_point, extra_point, selection_completed

        if event == cv2.EVENT_LBUTTONDOWN:
            start_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            end_point = (x, y)
            cv2.rectangle(clone, start_point, end_point, (0, 255, 0), 2)
            cv2.imshow("Image", clone)

        elif event == cv2.EVENT_RBUTTONDOWN:
            extra_point = (x, y)
            cv2.circle(clone, extra_point, 3, (0, 0, 255), -1)
            cv2.imshow("Image", clone)

    # Create a window and set the mouse callback function
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image", mouse_callback)

    # Start the selection loop
    while not selection_completed:
        cv2.imshow("Image", clone)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            # Reset the selection if 'r' key is pressed
            clone = image.copy()
            start_point = None
            end_point = None
            extra_point = None

        elif key == ord("c"):
            # Complete the selection if 'c' key is pressed
            if start_point is not None and end_point is not None:
                selection_completed = True

    # Close the OpenCV windows
    cv2.destroyAllWindows()

    # If the rectangle or the extra point is not selected, return None
    if start_point is None or end_point is None or extra_point is None:
        return None

    # Collect coordinates of vertices
    top_left = (min(start_point[0], end_point[0]), min(start_point[1], end_point[1]))
    bottom_right = (max(start_point[0], end_point[0]), max(start_point[1], end_point[1]))

    # Return the selected coordinates
    return top_left, bottom_right, extra_point


# Load data and models outside the streamlit loop
filename = "./myntradataset/styles.csv"
df = pd.read_csv(filename, on_bad_lines="skip")
available_ids = os.listdir("./myntradataset/images")
available_ids = [int(x.replace(".jpg", "")) for x in available_ids]
df = df[df.id.isin(available_ids)]
df = df.dropna(subset='productDisplayName')

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

knn_vectors = torch.load("image_tensor.pt")
knn_labels = df.id.astype(str).tolist()

vectors_np = knn_vectors.numpy()

dimension = vectors_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors_np)

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
sam_checkpoint ="./sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

@st.cache(allow_output_mutation=True)  # Caching the segmentation mask computation
def get_segmentation_mask(image_rgb):
    predictor.set_image(image_rgb)
    top_left, bottom_right, extra_point = select_rectangle_with_point(image_rgb)
    xyxy = list(top_left)
    temp = list(bottom_right)
    xyxy[len(xyxy):] = temp
    input_box = np.array(xyxy)
    input_point = np.array(list([extra_point]))
    input_label = np.array([1])
    
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=False,
    )
    int_mask = masks.astype(int)
    
    return int_mask

@st.cache(allow_output_mutation=True)  # Caching the image embedding and nearest neighbor search
def get_neighbors(custom_embedding):
    k = 20
    D, I = index.search(custom_embedding.detach().numpy(), k)
    neighbor_labels = [knn_labels[i] for i in I[0]]
    return neighbor_labels

uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    image = np.array(Image.open(uploaded_image))
    st.image(image, caption="Uploaded Image")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    int_mask = get_segmentation_mask(image_rgb)

    width = image.shape[0]
    height = image.shape[1]
    for i in range(width):
        for j in range(height):
            if int_mask[0][i][j] == 0:
                image[i][j] = (255, 255, 255)
    
    st.image(image, caption="Segmented Image")

    custom_input = processor(text=[''], images=image, return_tensors="pt", padding=True)
    custom_output = model(**custom_input)
    custom_embedding = custom_output.image_embeds

    neighbor_labels = get_neighbors(custom_embedding)

    directory = "./myntradataset/images/"
    num_columns = 5  # Number of columns in the grid
    num_images = len(neighbor_labels)
    num_rows = (num_images + num_columns - 1) // num_columns

    with st.container():
        for row in range(num_rows):
            cols = st.columns(num_columns)
            for col in range(num_columns):
                idx = row * num_columns + col
                if idx < num_images:
                    label = neighbor_labels[idx]
                    file_path = os.path.join(directory, f"{label}.jpg")
                    cols[col].image(file_path, use_column_width=True)
                    cols[col].write(df[df['id'] == int(label)].productDisplayName.iloc[0])
                else:
                    break
