import torch
from PIL import Image
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from paths import sam_checkpoint





# def select_rectangle_with_point(image):
#     # Make a copy of the image for display
#     clone = image.copy()
#
#     # Initialize variables for storing coordinates and selection status
#     start_point = None
#     end_point = None
#     extra_point = None
#     selection_completed = False
#
#     def mouse_callback(event, x, y, flags, param):
#         nonlocal start_point, end_point, extra_point, selection_completed
#
#         if event == cv2.EVENT_LBUTTONDOWN:
#             start_point = (x, y)
#
#         elif event == cv2.EVENT_LBUTTONUP:
#             end_point = (x, y)
#             cv2.rectangle(clone, start_point, end_point, (0, 255, 0), 2)
#             cv2.imshow("Image", clone)
#
#         elif event == cv2.EVENT_RBUTTONDOWN:
#             extra_point = (x, y)
#             cv2.circle(clone, extra_point, 3, (0, 0, 255), -1)
#             cv2.imshow("Image", clone)
#
#     # Create a window and set the mouse callback function
#     cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
#     cv2.setMouseCallback("Image", mouse_callback)
#
#     # Start the selection loop
#     while not selection_completed:
#         cv2.imshow("Image", clone)
#         key = cv2.waitKey(1) & 0xFF
#
#         if key == ord("r"):
#             # Reset the selection if 'r' key is pressed
#             clone = image.copy()
#             start_point = None
#             end_point = None
#             extra_point = None
#
#         elif key == ord("c"):
#             # Complete the selection if 'c' key is pressed
#             if start_point is not None and end_point is not None:
#                 selection_completed = True
#
#     # Close the OpenCV windows
#     cv2.destroyAllWindows()
#
#     # If the rectangle or the extra point is not selected, return None
#     if start_point is None or end_point is None or extra_point is None:
#         return None
#
#     # Collect coordinates of vertices
#     top_left = (min(start_point[0], end_point[0]), min(start_point[1], end_point[1]))
#     bottom_right = (max(start_point[0], end_point[0]), max(start_point[1], end_point[1]))
#
#     # Return the selected coordinates
#     return top_left, bottom_right, extra_point
























# def get_clicked_point(image):
#     # Load the image using OpenCV
#
#     # Create a copy of the image for display
#     display_image = image.copy()
#     image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
#
#     # Flag to indicate if a point has been captured
#     point_captured = False
#
#     # Variables to store the clicked coordinates
#     clicked_x = None
#     clicked_y = None
#
#     def click_callback(event, x, y, flags, param):
#         nonlocal point_captured, clicked_x, clicked_y
#
#         # Check if the left mouse button is pressed
#         if event == cv2.EVENT_LBUTTONDOWN and not point_captured:
#             # Store the clicked coordinates
#             clicked_x = x
#             clicked_y = y
#
#             # Print the coordinates
#             print(f"Clicked coordinates: x={clicked_x}, y={clicked_y}")
#
#             # Draw a circle at the clicked point
#             cv2.circle(image_rgb, (clicked_x, clicked_y), 5, (0, 0, 255), -1)
#             cv2.imshow("Image", image_rgb)
#
#             # Set the flag to indicate a point has been captured
#             point_captured = True
#
#     # Create a window and set the callback function
#     cv2.namedWindow("Image")
#     cv2.setMouseCallback("Image", click_callback)
#
#     # Display the image
#     cv2.imshow("Image", image_rgb)
#
#     # Wait for a point to be captured or 'Esc' key is pressed
#     while not point_captured:
#         key = cv2.waitKey(1) & 0xFF
#         if key == 27:  # 'Esc' key
#             break
#
#     # Close all windows
#     cv2.destroyAllWindows()
#
#     # Return the clicked coordinates if a point was captured, otherwise return None
#     if point_captured:
#         return [(clicked_x, clicked_y)]
#     else:
#         return None

#
# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30/255, 144/255, 255/255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)
#
# def show_points(coords, labels, ax, marker_size=375):
#     pos_points = coords[labels==1]
#     neg_points = coords[labels==0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#
# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))








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





def color_change_image(image, masks, color):

    int_mask = masks.astype(int)

    width = image.shape[0]
    height = image.shape[1]

    #Fix a base color
    startCol = get_avg_color(image, masks)

    fixColor = []
    fixColor.append(str(color[0]))
    fixColor.append(str(color[1]))
    fixColor.append(str(color[2]))


    if color[0]==255 and color[1]==0 and color[2]==0:
        fixColor[1] ="65"
    if color[0]==0 and color[1]==255 and color[2]==0:
        fixColor[0] = "65"
    if color[0]==0 and color[1]==0 and color[2]==255:
        fixColor[1] = "65"

    newColor = (int(fixColor[0]),int(fixColor[1]),int(fixColor[2]))

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
