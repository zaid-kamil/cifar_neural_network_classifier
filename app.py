import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO


def get_name(idx):
    classes = ['airplane','automobile','bird','cat','deer','dog',
                'frog','horse','ship','truck']
    return classes[idx]

def predict_image(imgarr, path_of_model='cifar_10_nn_model.h5'):
    model = tf.keras.models.load_model(path_of_model)
    result = model.predict(np.array([imgarr]))
    idx = result.argmax()
    return get_name(idx)

def get_web_image(url,size=(32,32)):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    thumb = img.resize(size=size)
    return thumb,img
st.title("CIFAR 10 Image Predictiom")

with st.form("ImageUploadForm"):
    imgUrl = st.text_input("put an image url")
    st.form_submit_button("Guess the image content")

if imgUrl:
    thumb,img = get_web_image(imgUrl)
    caption = predict_image(np.array(thumb))
    st.image(img,caption=caption)
    st.success(f"img contains {caption}")


