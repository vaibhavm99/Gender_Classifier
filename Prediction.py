# 0 = female  1 = male
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
model = load_model("model.h5")
x = plt.imread("2.png")
height = 220
width = 220
dim = (width, height)
img = cv2.resize(x, dim, interpolation=cv2.INTER_LINEAR)
if img.shape != (220,220):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
z = np.zeros((1,220,220))
z[0] = g
gen = model.predict(z)
gender = ["Female","Male"]
print(gen)
if(gen < 0.5):
    gen = 0
else:
    gen = 1
print(gen)
