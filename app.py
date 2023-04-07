import streamlit as st
import io
import numpy as np
from PIL import Image
import tensorflow as tf
import efficientnet.tfkeras as efn

st.set_page_config(page_title='Virtue', page_icon = 'assets/images/logo.png')
# Title and Description
st.title("Virtue Image")
st.write("Just Upload your Plant's Leaf Image and get predictions if the plant is healthy or not") 




model = tf.keras.models.load_model('model.h5')


uploaded_file = st.file_uploader('Choose your image', type=['png', 'jpg'])

predictions_map = {0:'is healthy', 1:'hast Multiple Diseases', 2:'has rust', 3:'has scab'}

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(image, use_column_width=True)
    
    resized_image = np.array(image.resize((512,512)))/255. # Resize image and divide pixel number by 255. for having values between 0 and 1 (normalize it)
    
    image_batch = resized_image[np.newaxis, :, :, :]
    
    predictions_arr = model.predict(image_batch)
    
    predictions = np.argmax(predictions_arr)

    result_text = f'The plant leaf {predictions_map[predictions]}'

    if predictions == 0:
        st.success(result_text)
    else:
        st.error(result_text)





    
