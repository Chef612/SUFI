from keras.models import load_model, model_from_json
from PIL import Image
import json
import cv2
import numpy as np 

# Load model from Json file 
json_file = open('model.json','r')
loaded_model = json_file.read()
json_file.close()

load_model = model_from_json(loaded_model)
load_model.load_weights('model.h5')
v=cv2.VideoCapture(0)
# Load Image 
image = Image.open('face_332.jpg') ## Test Image Path
image1 = Image.open('File_10,027.jpg')
im = image.resize((200,200))
im1 = image1.resize((200,200))

im = np.asarray(im)
im = np.reshape(im,(1,im.shape[0],im.shape[1],im.shape[2]))
im1 = np.asarray(im1)
im1 = np.reshape(im1,(1,im1.shape[0],im1.shape[1],im1.shape[2]))

# Make Prediction 
prediction = load_model.predict(im)
if prediction == 1:
  print('Real Face')
else:
  print('Fake Face')

 
prediction = load_model.predict(im1)
if prediction == 1:
  print('Real Face')
else:
  print('Fake Face')


