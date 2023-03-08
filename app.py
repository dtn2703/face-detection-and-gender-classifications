import streamlit as st

from src.data import *
from src.input import webcam_input

st.title("Face Detection - Gender Classifications")

st.sidebar.header('Options')
detection_model_name = st.sidebar.selectbox("Choose the style model: ", face_detection_model_name)

recognition_model_name = st.sidebar.selectbox("Choose the style model: ", gender_recognition_model_name)


webcam_input(detection_model_name, recognition_model_name)
