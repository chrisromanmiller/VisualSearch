import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor


sam_checkpoint ="./sam_vit_h_4b8939.pth"



def get_clicked_point(image):
    # Load the image using OpenCV

    # Create a copy of the image for display
    display_image = image.copy()
    image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)

    # Flag to indicate if a point has been captured
    point_captured = False

    # Variables to store the clicked coordinates
    clicked_x = None
    clicked_y = None

    def click_callback(event, x, y, flags, param):
        nonlocal point_captured, clicked_x, clicked_y

        # Check if the left mouse button is pressed
        if event == cv2.EVENT_LBUTTONDOWN and not point_captured:
            # Store the clicked coordinates
            clicked_x = x
            clicked_y = y

            # Print the coordinates
            print(f"Clicked coordinates: x={clicked_x}, y={clicked_y}")

            # Draw a circle at the clicked point
            cv2.circle(image_rgb, (clicked_x, clicked_y), 5, (0, 0, 255), -1)
            cv2.imshow("Image", image_rgb)

            # Set the flag to indicate a point has been captured
            point_captured = True

    # Create a window and set the callback function
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", click_callback)

    # Display the image
    cv2.imshow("Image", image_rgb)

    # Wait for a point to be captured or 'Esc' key is pressed
    while not point_captured:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # 'Esc' key
            break

    # Close all windows
    cv2.destroyAllWindows()

    # Return the clicked coordinates if a point was captured, otherwise return None
    if point_captured:
        return [(clicked_x, clicked_y)]
    else:
        return None


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


def segment_anything_pick_object(uploaded_file):
    model_type = "vit_h"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    image_Seg = np.array(Image.open(uploaded_file))

    clicked_point = get_clicked_point(image_Seg)
    predictor = SamPredictor(sam)
    predictor.set_image(image_Seg)

    #This is where we'll want to put in Tanuj's code to get user inputs instead of these presets
    #So you'll need to map you're image and change the 125 and 200 to lie on top of whatever item you want to segment
    input_point = np.array(clicked_point)
    input_label = np.array([1])

    #Making Mask
    masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,)

    mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

    masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],
    multimask_output=False,)

    int_mask = masks.astype(int)

    #image_Seg = cv2.cvtColor(image_Seg, cv2.COLOR_YCrCb2BGR)

    # Extracting the width and height
    # of the image:
    width = image_Seg.shape[0]
    height = image_Seg.shape[1]
    for i in range(width):
        for j in range(height):
            # getting the current RGB value of pixel (i,j).
            if int_mask[0][i][j]==0:
                image_Seg[i][j]=(255,255,255)

    return image_Seg
