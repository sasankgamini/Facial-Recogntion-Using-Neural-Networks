import cv2
import os
import numpy as np
import sklearn
import sklearn.model_selection
import keras

datafolder='SomethingOrNothing'
data=[]
labels=[]
folders=['Something','Nothing']
for symbol in folders:
    path = os.path.join(datafolder,symbol)
    images = os.listdir(path)
    for eachImage in images:
        imgarray = cv2.imread(os.path.join(path,eachImage))
        data.append(imgarray)
        if symbol == "Something":
            labels.append(0)
        if symbol == "Nothing":
            labels.append(1)


data = np.array(data)
labels = np.array(labels)

print(len(data))
print(len(labels))

train_images,test_images,train_labels,test_labels=sklearn.model_selection.train_test_split(data,labels,test_size=0.1)
train_images=train_images/255
test_images=test_images/255

print(train_images.shape)
print(test_images.shape)
'''
#building the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=())
'''
