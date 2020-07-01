# 0 = female  1 = male
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
#Timing the code
import time
start = time.time()
#Preprocessing Data
# Y = labels
y = loadmat("labels.mat")
y = y["y"]
y = y.reshape(-1,1)
# X = training data
size = 300
y_final = y[:size]
names = []
for i in range(size):
    names.append(str(i+1)+'.mat')
X = []
for i in range(size):
    temp = loadmat(names[i])
    temp = temp["s"]
    print("Appending Image => ",i+1,"   ",((i+1)/size)*100,"%")
    X.append(temp)
pre = time.time()
print("Preprocessing time = ", pre - start, " seconds")

#Saving the data
#import pickle
#with open('train.pickle', 'wb') as f:
 #   pickle.dump([X, y], f)    
# To load the Variables, use this:
#    with open('train.pickle', 'rb') as f:
#       X_train, y_train = pickle.load(f)

#Resizing the image
import cv2
height = 220
width = 220
dim = (width, height)
X_final = []
for i in range(size):
    print("Resizing Image => ",i+1,"   ",((i+1)/size)*100,"%")
    res = cv2.resize(X[i], dim, interpolation=cv2.INTER_LINEAR)
    if res.shape != (220,220):
        g = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    X_final.append(g)
X_final = np.stack(X_final)    
#Handling missing values
for i in range(size):
    if (np.isnan(y_final[i])):
        X_final[i] = 0
        y_final[i] = 0
#Splitting into test and train
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_final,y_final,test_size = 0.1)
#Normalizing the data
from sklearn.preprocessing import normalize
#X_train = X_train/255.0
#X_test = X_test/255.0
#Model Definition
model = Sequential()
model.add(Conv1D(64,kernel_size = (3), padding = 'same',activation = 'relu', input_shape = (220,220)))
model.add(MaxPooling1D(pool_size = (2)))

model.add(Conv1D(64,kernel_size = (3),padding = 'same',activation = 'relu'))
model.add(MaxPooling1D(pool_size = (2)))

model.add(Conv1D(64,kernel_size = (5),padding = 'same',activation = 'relu'))
model.add(MaxPooling1D(pool_size = (2)))

model.add(Conv1D(64,kernel_size = (5),padding = 'same',activation = 'relu'))
model.add(MaxPooling1D(pool_size = (2)))

model.add(Conv1D(64,kernel_size = (7),padding = 'same',activation = 'relu'))
model.add(MaxPooling1D(pool_size = (2)))

model.add(Conv1D(64,kernel_size = (7),padding = 'same',activation = 'relu'))
model.add(MaxPooling1D(pool_size = (2)))

model.add(Conv1D(64,kernel_size = (9),padding = 'same',activation = 'relu'))
model.add(MaxPooling1D(pool_size = (2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Dense(64))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print("Model Defined. Training now.")
#Training the model
model.fit(X_train,y_train,batch_size = 32,epochs = 50)
print("Model trained!")
#Prediction
y_pred = model.predict(X_test)
y_pred[y_pred>=0.5] = 1
y_pred[y_pred<0.5] = 0
#Accuracy
from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)
print("R2 score = ", score)
x,results=model.evaluate(X_test,y_test)
print("Accuracy = ",results)
model.save("model.h5")
print("Execution time = ", time.time() - start, " seconds")
print("Preprocessing time = ", pre - start, " seconds")