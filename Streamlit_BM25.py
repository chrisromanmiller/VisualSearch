import streamlit as st
from combined_search import *
from segment_anything_pick_object import segment_anything_pick_object, select_rectangle_with_point_st
from displaying_images import display_images_ids_st




with st.sidebar:
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg'])
    selected = st.checkbox("Use Segment Anything")
    query = st.text_input("BM25 Search")
    query2 = st.text_input("Clip Search")
    button = st.button("Search")


if uploaded_file is not None:
    st.image(uploaded_file, width = 200)
    image = np.array(Image.open(uploaded_file))

    if selected:
        with st.container():
            cols = st.columns(2)
            with cols[0]:
                left = st.slider(label="left", min_value = 0, max_value = image.shape[1])
                top = st.slider(label="top", min_value = 0, max_value = image.shape[0])
                x_coord = st.slider(label="x_coord", min_value = 0, max_value = image.shape[1])
            with cols[1]:
                right = st.slider(label="right", min_value = 0, max_value = image.shape[1])
                bottom = st.slider(label="bottom", min_value = 0, max_value = image.shape[0])
                y_coord = st.slider(label="y_coord", min_value = 0, max_value = image.shape[0])
                
        image = select_rectangle_with_point_st(left, right, top, bottom, x_coord, y_coord, image)
        st.image(image, width = 200)



    if (button and selected):

        image_Seg = segment_anything_pick_object(left, right, image.shape[0]-1-top, image.shape[0]-1-bottom, x_coord, image.shape[0]-1-y_coord, uploaded_file)
        st.image(image_Seg, width = 200)
        st.text('Results:')
        ids, df = combined_search(image_Seg, query, query2)
        display_images_ids_st(ids, df)




if (button and not selected):
    st.text('Results:')
    ids, df = combined_search(uploaded_file, query, query2)
    display_images_ids_st(ids, df)
