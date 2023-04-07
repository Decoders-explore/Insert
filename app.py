import streamlit as st
import tensorflow as tf
import cv2

import time


from utils import *

st.set_page_config(page_title='Virtue', page_icon = 'assets/images/logo.png')
st.title("Virtue Image")



method = st.selectbox('Capture or Upload an Image', ('Upload Image', 'Capture Image'))

if method == 'Upload Image':
    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
else:
    image_file = st.camera_input("Capture Image")



