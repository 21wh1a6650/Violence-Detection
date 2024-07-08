# streamlit_app.py

import streamlit as st
import numpy as np
#import matplotlib.pyplot as plt
from inference import predict_video
from keras.models import load_model
import os
import cv2
from io import BytesIO
import time
from PIL import Image
 
SEQUENCE_LENGTH = 16
NUM_DISPLAY_FRAMES = 10
RESIZE_WIDTH = 50  # Adjust as needed
RESIZE_HEIGHT = 50

st.markdown(
    """
    <style>
    .stApp {
        background-image:url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRRIvABHBkiXjHCUC9ykbz5WxawZYDnbh6T0SksQ04-PHeW4XDJYPzG-5Vvm4hKdgaRm7k&usqp=CAU');
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Violence Detector")
uploaded_file = st.file_uploader("Upload a video file (MP4)", type=["mp4"])

if uploaded_file:
    if uploaded_file.type == 'video/mp4':
        st.write("Processed Video:")
        st.video(uploaded_file)

        model_path = os.path.join(r"C:\Users\sudes\Downloads\trained_model.h5")  # Replace with the path to your trained model
        model = load_model(model_path)
        # Define a CSS style to center the button
        style = """
            <style>
            .stButton {
              margin: 0 auto;
              text-align: center; 
            }
            </style>
            """
            # Add the CSS style to your app
        st.markdown(style, unsafe_allow_html=True)

            # Add the button
        if st.button("Predict"):
            # Add the CSS style to your app
            #with st.spinner('Processing...'):
              #time.sleep(2)
            st.success('Done')
            bar = st.progress(50)
            time.sleep(5)
            bar.progress(100)   
            # Save the uploaded video to a temporary file in the same directory as the script
            temp_file_path = "temp_uploaded_video.mp4"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.read())  # Use .read() to get the content of BytesIO

            # Predict violence using the saved video
            result = predict_video(temp_file_path, SEQUENCE_LENGTH)
            if result=="NonViolence":
                 result="Non Violence"
                 st.markdown('<div style="font-family: Times New Roman, serif; text-align: center;color: green; font-size: 24px;">Non Violence Detected.<br>Keep spreading peace and happiness!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="color: red;text-align: center;font-family: Times New Roman, serif; font-size: 24px;">Violence Detected.<br>Let\'s work towards a peaceful world.</div>', unsafe_allow_html=True)
            
             # st.write(result," detected")
            #######################################################
            # Center the image horizontally and set the image width and height
            if result == "Violence":
                col1, col2, col3 = st.columns([1, 1, 1])
                # Display the image in the middle column
                col2.image("Violence.jpg")
            else:
                col1, col2, col3 = st.columns([1, 1, 1])
                # Display the image in the middle column
                col2.image("NonViolence.jpg")
         ########################################################3  
            os.remove(temp_file_path)
    else:
        st.warning("Please upload an MP4 file.")
