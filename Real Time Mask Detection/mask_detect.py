import cv2
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from keras.preprocessing.image import array_to_img
import numpy as np


mapping = {0: "incorrect mask", 1: "with mask", 2: "without mask"}


st.title("Real Time Mask Detection By Beket")
FRAME_WINDOW = st.image([])
cam = cv2.VideoCapture(1)
text = st.empty()
model = keras.models.load_model('mask_model.h5')
while True:
    ret, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

    input_arr = np.array([frame])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    prediction_label = mapping[predictions.argmax()]
    text.markdown(f"### **{prediction_label.upper()}**")

else:
    st.write('Stopped')