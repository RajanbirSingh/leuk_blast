# from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import streamlit as st
from keras.models import model_from_json
import cv2
# import glob
import pandas as pd
# import matplotlib.image as mpimg 
# import matplotlib.pyplot as plt 
from PIL import Image, ImageOps
import tensorflow as tf

@st.cache(allow_output_mutation=True)
# def load_model():
#   # json_file = open('model1.json', 'r')
#   # loaded_model_json = json_file.read()
#   # json_file.close()
#   # loaded_model = model_from_json(loaded_model_json)
#   # load weights into new model
#   loaded_model.load_weights("model1.h5")
#   return loaded_model
def load_model():
  return tf.keras.models.load_model("models.hdf5")

loaded_model = load_model()

      
def predict(img):
    loaded_model = load_model()
    image = np.asarray(img)
    image = image/255.0
    image = np.reshape(image,[1,400,400,3])

    classes = loaded_model.predict_classes(image)

    return classes