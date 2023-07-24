import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import streamlit as st
from paths import images_path





def display_images_and_titles(filepaths, titles):
    # Create a 4x5 grid of subplots
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))

    # Loop through the filepaths and display images in the subplots
    for i, ax in enumerate(axes.flatten()):
        if i < len(filepaths):
            filepath = filepaths[i]
            image = plt.imread(filepath)
            ax.imshow(image)
            ax.axis('off')

            # Assign a title to each subplot
            ax.set_title(f"{titles[i]}", fontsize = 7)
        else:
            ax.axis('off')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()



def display_images_and_titles_from_ind(ind, df):
    directory = images_path

    titles = df.productDisplayName.tolist()

    result_subset = df.iloc[ind]
    filenames = result_subset.id.astype(str).tolist()
    filepaths = [f"{os.path.join(directory,filename)}.jpg"  for filename in filenames]

    np_arr = np.array(titles)
    result_titles = np_arr[ind].tolist()

    display_images_and_titles(filepaths, result_titles)




def display_images_titles_from_ids(ids, df):

    directory = images_path

    df_new_index = df.copy()
    df_new_index['id'] = df_new_index['id'].astype(str)
    df_new_index.set_index("id",inplace = True)
    df_new_index = df_new_index.loc[ids]
    filenames = df_new_index.index.tolist()
    filepaths = [f"{os.path.join(directory,filename)}.jpg"  for filename in filenames]
    titles =  df_new_index.productDisplayName.tolist()

    display_images_and_titles(filepaths, titles)








def display_single_image_id_st(single_id, directory, column):
    column.image(directory + single_id + ".jpg", use_column_width=True)





def display_images_ids_st(ids, df):

    directory = images_path

    num_columns = 5  # Number of columns in the grid
    num_images = len(ids)
    num_rows = (num_images + num_columns - 1) // num_columns

    with st.container():
        for row in range(num_rows):
            cols = st.columns(num_columns)
            for col in range(num_columns):
                idx = row * num_columns + col
                if idx < num_images:
                    display_single_image_id_st(ids[idx], directory, cols[col])
                    cols[col].write(df[df['id'] == int(ids[idx])].productDisplayName.iloc[0])
                else:
                    break
