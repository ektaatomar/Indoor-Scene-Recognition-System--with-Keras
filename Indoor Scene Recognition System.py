# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:44:23 2018

@author: anagh
'''"""
''''
from skimage.transform import rotate,rescale
img_array = cv2.imread("C:/Users/anagh/Documents/Anagha_UTA/Fall 2018/DS/project/extra/bakery2.jpeg",cv2.IMREAD_GRAYSCALE)
rot = rotate(img_array, angle=90, mode='reflect')
new_array = cv2.resize(rot, (100,100))
new_array=new_array.reshape(-1,100,100,1)
plt.imshow(new_array, cmap='gray')

img_array = cv2.imread("C:/Users/anagh/Documents/Anagha_UTA/Fall 2018/DS/project/extra/bakery2.jpeg",cv2.IMREAD_GRAYSCALE)
scaled = rescale(img_array, scale=50, mode='constant')
new_array = cv2.resize(scaled, (100,100))
new_array=new_array.reshape(-1,100,100,1)
plt.imshow(new_array, cmap='gray')
'''


import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]    

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import keras

from tensorflow.keras.callbacks import TensorBoard
import time



datadir = "C:/Users/anagh/Documents/Anagha_UTA/Fall 2018/DS/project/Images"
categories = ["airport_inside","artstudio","auditorium","bakery","bar","bathroom","bedroom","bookstore","closet",
              "dentaloffice","elevator","florist","garage","gym","hairsalon","hospitalroom","library","livingroom",
              "mall","office","poolinside"]


img_size = 100      # resize all the images to one size

training_data=[]
def create_training_data():
    for category in categories:
        path = os.path.join(datadir, category) # path to categories
        class_num = categories.index(category) 
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) #read image and convert to gray scale
                new_array = cv2.resize(img_array,(img_size,img_size))
                random_rotation(new_array)
                training_data.append([new_array,class_num])
                random_noise(new_array)
                training_data.append([new_array,class_num])
                horizontal_flip(neew_array)
                training_data.append([new_array,class_num])  
            except Exception as e:
                    pass
create_training_data()

import random
random.shuffle(training_data)

X = []
y = []
for features,label in training_data:
    X.append(features)
    y.append(label)
    
X=np.array(X).reshape(-1,img_size,img_size,1)  #(cannot pass list directly, -1=(calculates the array size), size,1=gray scale)
class_num=keras.utils.np_utils.to_categorical(y,num_classes=len(categories))   #one-hot encoder for cateorical values

import tensorflow as tf
from keras import models
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D

#normalize data(scaling)

X=X/255.0

#build model

dense_layers=[4]
layer_sizes=[128]
conv_layers=[2]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            name = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer,layer_size,dense_layer,int(time.time()))
            print(name)
            tensorboard = TensorBoard(log_dir='C:/Users/anagh/Desktop/logs{}'.format(name))
            model = models.Sequential()
            
            model.add(Conv2D(layer_size,(3,3), input_shape = X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))
            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size,(3,3), input_shape = X.shape[1:]))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Flatten())
            for l in range(dense_layer):
                model.add(Dense(layer_size, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(len(categories), activation='softmax'))   
            model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
            model.fit(X, class_num, epochs=5, batch_size=32,validation_split=0.2,callbacks=[tensorboard])

'''
#predicting
filepath="C:/Users/anagh/Documents/Anagha_UTA/Fall 2018/DS/project/extra/"
def prepare(img):                                      #preprocessing the new image
    img_array = cv2.imread(filepath+str(img),cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (100,100))
    return new_array.reshape(-1,100,100,1)

prediction = model.predict([new_array])
print(categories[np.argmax(prediction)])


img_array = cv2.imread("C:/Users/anagh/Documents/Anagha_UTA/Fall 2018/DS/project/extra/airport1.jpg",cv2.IMREAD_GRAYSCALE)
new_array = cv2.resize(img_array, (100,100))
new_array=new_array.reshape(-1,100,100,1)
plt.imshow(new_array, cmap='gray')

img="airport1.jpg"
'''