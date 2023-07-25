import streamlit as st
import webcolors as wc
from combined_search import *
from segment_anything_pick_object import *
from displaying_images import display_images_ids_st




with st.sidebar:
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg'])
    use_sam = st.checkbox("Use Segment Anything")

    if use_sam:
        pattern_change = st.selectbox("Pattern Change", ("keep original","gingham","chevron","damask","dogstooth","check","glen plaid", "herringbone", "horizontal thick stripes", "horizontal thin stripes", "houndstooth", "ikat", "jacobean", "leather","leopard print", "matelasse", "ogee", "paisley", "plaid", "polka dot", "quarterfile", "toile", "vertical thick stripes", "vertical thin stripes"))
        side_cols = st.columns(2)
        with side_cols[0]:
            color_change = st.color_picker("Color Change")
            rgb = wc.hex_to_rgb(color_change)
        with side_cols[1]:
            use_color_change = st.checkbox("Use Color Change")

    query = st.text_input("BM25 Search")
    query2 = st.text_input("Clip Search")
    button = st.button("Search")


if uploaded_file is not None:
    st.image(uploaded_file, width = 200)
    image = np.array(Image.open(uploaded_file))

    if use_sam:
        with st.container():
            cols = st.columns(2)
            with cols[0]:
                left = st.slider(label="left", min_value = 0, max_value = image.shape[1])
                right = st.slider(label="right", min_value = 0, max_value = image.shape[1])
                x_coord = st.slider(label="x_coord", min_value = 0, max_value = image.shape[1])
            with cols[1]:
                bottom = st.slider(label="bottom", min_value = 0, max_value = image.shape[0])
                top = st.slider(label="top", min_value = 0, max_value = image.shape[0])
                y_coord = st.slider(label="y_coord", min_value = 0, max_value = image.shape[0])

        image = select_rectangle_with_point_st(left, right, top, bottom, x_coord, y_coord, image)
        st.image(image, width = 200)



    if (button and use_sam and not use_color_change):
        image_Seg, masks = segment_anything_pick_object(left, right, image.shape[0]-1-top, image.shape[0]-1-bottom, x_coord, image.shape[0]-1-y_coord, uploaded_file)
        if (pattern_change != "keep original"):
            pattern_change_image(image_Seg, masks, pattern_change)
        st.image(image_Seg, width = 200)
        st.text('Results:')
        ids, df = combined_search(image_Seg, query, query2)
        display_images_ids_st(ids, df)
    
    
    if (button and use_sam and use_color_change):
        image_Seg, masks = segment_anything_pick_object(left, right, image.shape[0]-1-top, image.shape[0]-1-bottom, x_coord, image.shape[0]-1-y_coord, uploaded_file)
        if (pattern_change != "keep original"):
            pattern_change_image(image_Seg, masks, pattern_change)
        color_change_image(image_Seg, masks, rgb)
        st.image(image_Seg, width = 200)
        st.text('Results:')
        ids, df = combined_search(image_Seg, query, query2)
        display_images_ids_st(ids, df)




if (button and not use_sam):
    st.text('Results:')
    ids, df = combined_search(uploaded_file, query, query2)
    display_images_ids_st(ids, df)
