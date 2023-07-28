import torch
from PIL import Image
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from paths import sam_checkpoint






# This function can pick out an object from an image (last argument) given coordinates for a rectangle (first 4 arguments)
# and a point to select the object of interest (arguments 5 and 6) using SAM
# The function returns the segmented image as well as the masks from SAM to do image manipulation

def segment_anything_pick_object(left, right, top, bottom, x_coord, y_coord, uploaded_image):

    model_type = "vit_h"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    if uploaded_image is not None:
        image = np.array(Image.open(uploaded_image))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if (image.shape[2] == 4):
            image = image[:, :, 0:3]

        predictor.set_image(image_rgb)

        top_left = (left, top)
        bottom_right = (right, bottom)
        extra_point = (x_coord, y_coord)



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

        width = image.shape[0]
        height = image.shape[1]
        for i in range(width):
            for j in range(height):
                # getting the current RGB value of pixel (i,j).
                if int_mask[0][i][j]==0:
                    image[i][j]=(255,255,255)

    return image, masks






# This function gets the average color of the selected mask from an image that has been processed with SAM

def get_avg_color(image, masks):

    int_mask = masks.astype(int)

    width = image.shape[0]
    height = image.shape[1]

    # Extracting the width and height of the image along with original colors and finding the average
    colList=[]
    for i in range(width):
        for j in range(height):
            # getting the current RGB value of pixel (i,j).
            if int_mask[0][i][j]==1:
                colList.append(image[i][j])
    L=len(colList)
    m1=0
    m2=0
    m3=0
    for i in colList:
        m1+=i[0]
        m2+=i[1]
        m3+=i[2]
    AvgCol = [(m1/L, m2/L, m3/L)]

    return AvgCol






# Scales values linearly from a list according to min_val and max_val

def scale_list(values, min_val, max_val):
    # Convert the list to a NumPy array
    arr = np.array(values)

    # Find the minimum and maximum values in the array
    arr_min = np.min(arr)
    arr_max = np.max(arr)

    # Scale the values to be between min_val and max_val
    scaled_arr = (arr - arr_min) * (max_val - min_val) / (arr_max - arr_min) + min_val

    # Convert the scaled array back to a list
    scaled_list = scaled_arr.tolist()

    return scaled_list






# This function changes the color of an image using SAM's masks. Returns altered image

def color_change_image(image, masks, color):

    int_mask = masks.astype(int)

    width = image.shape[0]
    height = image.shape[1]

    #Fix a base color
    startCol = get_avg_color(image, masks)

    

    newColor = color
    
    #How different is this base color from target color?
    dif1 = startCol[0][0] - newColor[0]
    dif2 = startCol[0][1] - newColor[1]
    dif3 = startCol[0][2] - newColor[2]

    #Make blank list for new color values
    newColR=[]
    newColG=[]
    newColB=[]

    #Make lists of the shifted colors from old to new RGB values
    for i in range(width):
        for j in range(height):
            if int_mask[0][i][j]==1:
                newColR.append(image[i][j][0]-dif1)
                newColG.append(image[i][j][1]-dif2)
                newColB.append(image[i][j][2]-dif3)

    #Rescale each color list to make sure it's in appropriate RGB value range
    minR = max(min(newColR), 0)
    maxR = min(max(newColR), 255)
    minG = max(min(newColG), 0)
    maxG = min(max(newColG), 255)
    minB = max(min(newColB), 0)
    maxB = min(max(newColB), 255)
    fixedColR = scale_list(newColR, minR,maxR)
    fixedColG = scale_list(newColG, minG,maxG)
    fixedColB = scale_list(newColB, minB,maxB)


    #Go through all pixels in the picture and set them to their new color
    n=0
    for i in range(width):
        for j in range(height):
            # Is this picture on the segment we're recoloring?
            if int_mask[0][i][j]==1:
                image[i][j][0] = fixedColR[n]
                image[i][j][1] = fixedColG[n]
                image[i][j][2] = fixedColB[n]
                n+=1

    return image






# This function changes the patter of an image using SAM's masks. Returns altered image

def pattern_change_image(image, masks, pattern_str):

    int_mask = masks.astype(int)

    width = image.shape[0]
    height = image.shape[1]

    pattern_name = pattern_str.replace(" ", "_")
    pattern_path = './Fabric_Swatches/' + pattern_name + ".jpeg"
    pattern = Image.open(pattern_path)
    print(f"Original size : {image.size}") # 5464x3640

    pattern_resized = pattern.resize((height, width))
    pattern_resized.save(pattern_path)

    pattern = cv2.imread(pattern_path)
    pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2RGB)

    #Fix a base color
    startCol = get_avg_color(image, masks)

    dif = []

    #How different is this base color from target color?
    for i in range(width):
        for j in range(height):
            if int_mask[0][i][j]==1:
                dif1 = startCol[0][0] - pattern[i][j][0]
                dif2 = startCol[0][1] - pattern[i][j][1]
                dif3 = startCol[0][2] - pattern[i][j][2]
                #dif1 = pattern[i][j][0]
                #dif2 =  pattern[i][j][1]
                #dif3 =  pattern[i][j][2]
                dif.append((dif1,dif2,dif3))

    #Make blank list for new color values
    newColR=[]
    newColG=[]
    newColB=[]

    #Make lists of the shifted colors from old to new RGB values
    c= 0
    for i in range(width):
        for j in range(height):
            if int_mask[0][i][j]==1:
                newColR.append(image[i][j][0]-dif[c][0])
                newColG.append(image[i][j][1]-dif[c][1])
                newColB.append(image[i][j][2]-dif[c][2])
                c+=1

    #Rescale each color list to make sure it's in appropriate RGB value range
    minR = max(min(newColR), 0)
    maxR = min(max(newColR), 255)
    minG = max(min(newColG), 0)
    maxG = min(max(newColG), 255)
    minB = max(min(newColB), 0)
    maxB = min(max(newColB), 255)
    fixedColR = scale_list(newColR, minR,maxR)
    fixedColG = scale_list(newColG, minG,maxG)
    fixedColB = scale_list(newColB, minB,maxB)


    #Go through all pixels in the picture and set them to their new color
    n=0
    for i in range(width):
        for j in range(height):
            # Is this picture on the segment we're recoloring?
            if int_mask[0][i][j]==1:
                image[i][j][0] = fixedColR[n]
                image[i][j][1] = fixedColG[n]
                image[i][j][2] = fixedColB[n]
                n+=1
    return image






# This function creates an image with a rectangle and a point to use for the user interface

def select_rectangle_with_point_st(left, right, top, bottom, x_coord, y_coord, image):

    top = image.shape[0]-1-top
    bottom = image.shape[0]-1-bottom
    y_coord = image.shape[0]-1-y_coord

    if (image.shape[2] == 4):
        image = image[:, :, 0:3]

    green = np.array([[[0, 255, 0]]])
    red = np.array([[[255, 0, 0]]])

    left_shape = image[top:bottom, left:left+10, 0:3].shape
    right_shape = image[top:bottom, right-10:right, 0:3].shape
    top_shape = image[top:top+10, left:right, 0:3].shape
    bottom_shape = image[bottom-10:bottom, left:right, 0:3].shape

    ex_pnt_shape = image[y_coord-10:y_coord+10, x_coord-10:x_coord+10, 0:3].shape

    image[top:top+10, left:right, 0:3] = np.repeat(np.repeat(green, right-left, axis=1), top_shape[0], axis=0)
    image[bottom-10:bottom, left:right, 0:3] = np.repeat(np.repeat(green, right-left, axis=1), bottom_shape[0], axis=0)
    image[top:bottom, left:left+10, 0:3] = np.repeat(np.repeat(green, left_shape[1], axis=1), bottom-top, axis=0)
    image[top:bottom, right-10:right, 0:3] = np.repeat(np.repeat(green, right_shape[1], axis=1), bottom-top, axis=0)

    image[y_coord-10:y_coord+10, x_coord-10:x_coord+10, 0:3] = np.repeat(np.repeat(red, ex_pnt_shape[1], axis=1), ex_pnt_shape[0], axis=0)

    return image
