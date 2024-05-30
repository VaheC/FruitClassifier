import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
# import imutils
# import easyocr
# import os
from fastai.vision.all import *
import pathlib
import platform
# import shutil


system_platform = platform.system()
if system_platform == 'Windows': pathlib.PosixPath = pathlib.WindowsPath


def main():
    st.title("Fruit Classifier")

    # Use st.camera to capture images from the user's camera
    img_file_buffer = st.camera_input(label='Please, take a photo of a fruit', key='fruit')

    # Check if an image is captured
    if img_file_buffer is not None:
        # Convert the image to a NumPy array
        # image = Image.open(img_file_buffer)
        # image_np = np.array(image)
        # resized_image = cv2.resize(image_np, (640, 640))
        # resized_image = resized_image.astype(np.uint8)
        # resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        image = cv2.imread(img_file_buffer)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('fruit_image.jpg', image)

        model = load_learner()
        model_output = model.predict('fruit_image.jpg')

        prob_idx = model.dls.vocab.index(model_output[0])
        st.write(f'{model_output[0].title()} is depicted in the photo with {model_output[-1][prob_idx]:.4f} confidence.')

        st.session_state.pop("fruit")

if __name__ == "__main__":
    main()