import streamlit as st
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from combined_search import *





with st.sidebar:
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg'])
    query = st.text_input("BM25 Search")
    button = st.button("Search")

if uploaded_file is not None:
    st.image(uploaded_file, width = 200)

st.text('Results:')



if button:
    ids, df = combined_search(uploaded_file, query)
    display_images_ids_st(ids, df)
        


    








