#-*-coding: utf-8-*-
"""
Created on 2018/5/25 下午 01:09 

@author: Leon
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import glob
import os
from cnnstructure import build
from keras.optimizers import Adamax
from keras import backend as K

training_fruit_img = []
training_label = []
for dir_path in glob.glob("input/fruits-360/Training/*"):
    img_label = dir_path.split("\\")[-1]
    for image_path in glob.glob(os.path.join(dir_path,"*.jpg")):
        image = cv2.imread(image_path,cv2.IMREAD_COLOR)
        image = cv2.resize(image, (32, 32))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        training_fruit_img.append(image)
        training_label.append(img_label)
training_fruit_img = np.array(training_fruit_img)
training_label = np.array(training_label)

label_to_id = {v:k for k,v in enumerate(np.unique(training_label)) }
id_to_label = {v:k for k,v in label_to_id.items() }
print(id_to_label)
validation_fruit_img = []
validation_label = []
for dir_path in glob.glob("input/fruits-360/Validation/*"):
    img_label = dir_path.split("\\")[-1]
    for image_path in glob.glob(os.path.join(dir_path,"*.jpg")):
        image = cv2.imread(image_path,cv2.IMREAD_COLOR)
        image = cv2.resize(image, (32, 32))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        validation_fruit_img.append(image)
        validation_label.append(img_label)
validation_fruit_img = np.array(validation_fruit_img)
validation_label = np.array(validation_label)
print(training_label)
training_label_id = np.array([label_to_id[i] for i in training_label])
validation_label_id = np.array([label_to_id[i] for i in validation_label])
X_train,X_test = training_fruit_img, validation_fruit_img
Y_train,Y_test =training_label_id, validation_label_id
#mean(X) = np.mean(X_trai
X_train = X_train/255
X_test = X_test/255


#One Hot Encode the Output
Y_train = keras.utils.to_categorical(Y_train, 60)
Y_test = keras.utils.to_categorical(Y_test, 60)
print('Original Sizes:', X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

model = build()
model.compile(loss='categorical_crossentropy',
             optimizer = Adamax(),
             metrics=['accuracy'])

H = model.fit(X_train,
          Y_train,
          batch_size=128,
          epochs=10,
          verbose=1,
          validation_data=(X_test,Y_test),
         )
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = 10
plt.plot(np.arange(0, N), np.array(H.history["loss"]), label="train_loss")
plt.plot(np.arange(0, N), np.array(H.history["val_loss"]), label="val_loss")
plt.plot(np.arange(0, N), np.array(H.history["acc"]), label="train_acc")
plt.plot(np.arange(0, N), np.array(H.history["val_acc"]), label="val_acc")
plt.title("Training Loss and Accuracy on traffic-sign classifier")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("loss function.png")
plt.show()

model.save('fruit_model.h5')