##streamlit run app.py
# Core Pkgs
import streamlit as st 
import cv2
from PIL import Image,ImageEnhance
import numpy as np 
import os
import matplotlib.pyplot as plt
import heapq
from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model



@st.cache
def load_image(img):
	im = Image.open(img)
	
	return im

def main():
	"""Image  classifier"""
	#prediction_model = add_model()
	#model = load_model('model.h5')
	st.title("Human body parts Classifier")
	st.text("Build with Tensorflow and keras - Deep Learning")
	st.text("This app can classify Eyes, Ears, Hands and Legs")
	activities = ["classifier","About"]
	choice = st.sidebar.selectbox("Select Activty",activities)
	image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
	if image_file is not None:
		our_image = Image.open(image_file)
		#print(our_image)
	if choice == 'classifier':	
		if image_file is not  None:
			model = tf.keras.models.load_model("mobilenetmodel.h5")
			#print(model.summary())
			
			new_width  = 150
			new_height = 150
			our_image = our_image.resize((new_width, new_height), Image.ANTIALIAS)
			our_image = np.array(our_image) 
			st.image(our_image)
			our_image = np.expand_dims(our_image, axis = 0)
			classes = ["Ears","Eyes","Hands","Heart","Legs"]
			result = model.predict(our_image)
			#training_set.class_indices
			#st.write(result)
			st.write("The prediction is: "+classes[np.argmax(np.array(result))])
			
			

	elif choice == 'About':
		st.subheader("Human body parts classifier")
		st.markdown("Built with Streamlit by Jyothi")
		st.text("Jyothi Ramilla")
		st.success("https://www.linkedin.com/in/jyothiramilla/")
	

if __name__ == '__main__':
	
	main()