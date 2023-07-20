import streamlit as st
from combined_search import *
from segment_anything_pick_object import *




with st.sidebar:
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg'])
    selected = st.checkbox("Use Segment Anything")
    query = st.text_input("BM25 Search")
    #query2 = st.text_input("Clip Search")
    button = st.button("Search")


if uploaded_file is not None:
    st.image(uploaded_file, width = 200)




if (button and selected and uploaded_file is not None):
    image_Seg = segment_anything_pick_object(uploaded_file)
    st.image(image_Seg, width = 200)
    st.text('Results:')
    ids, df = combined_search(image_Seg, query)
    display_images_ids_st(ids, df)




if (button and not selected):
    st.text('Results:')
    ids, df = combined_search(uploaded_file, query)
    display_images_ids_st(ids, df)
