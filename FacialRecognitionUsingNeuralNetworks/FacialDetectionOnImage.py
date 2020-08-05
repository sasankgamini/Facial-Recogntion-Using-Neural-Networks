import cv2
import os
import numpy as np
import sklearn
import sklearn.model_selection
import keras
##facecascade=cv2.CascadeClassifier('haarcascade_frontalface.xml')
##
##image=cv2.imread('../../images/Stephen Curry/download-2.jpg')
##grayscale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
##face=facecascade.detectMultiScale(grayscale,1.1,5)
##for (fx,fy,fw,fh) in face:
##    cv2.rectangle(image,(fx,fy),(fx+fw,fy+fh),(0,0,0),2)
##    ROI = grayscale[fy:fy+fh, fx:fx+fw]
##print(ROI)
##cv2.imshow('face',image)
##cv2.waitKey()
##cv2.destroyAllWindows()
    

datafolder='../../images'
facecascade=cv2.CascadeClassifier('haarcascade_frontalface.xml')
data=[]
labels=[]
folders=['Kevin Durant','Stephen Curry']
for symbol in folders:
    path = os.path.join(datafolder,symbol)
    images=os.listdir(path)
    for eachImage in images:
        imgarray=cv2.imread(os.path.join(path,eachImage))
        grayscale=cv2.cvtColor(imgarray,cv2.COLOR_BGR2GRAY)
        face=facecascade.detectMultiScale(grayscale,1.1,5)
        for (fx,fy,fw,fh) in face:
            ROI=grayscale[fy:fy+fh, fx:fx+fw]
            ROI=cv2.resize(ROI,(50,50))
            data.append(ROI)
            if symbol == "Kevin Durant":
                labels.append(0)
            elif symbol == "Stephen Curry":
                labels.append(1)

print(labels)
print(len(data))
print(len(labels))

data=np.array(data)
labels=np.array(labels)

train_images,test_images,train_labels,test_labels=sklearn.model_selection.train_test_split(data,labels,test_size=0.1)
train_images=train_images/255
test_images=test_images/255

print(train_images.shape)
print(test_images.shape)

#building the model(blueprint/design)
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(50,50)),
    keras.layers.Dense(128,activation='relu'), #activation if it passes certian threshold(relu: rectified linear unit)
    keras.layers.Dense(2,activation='softmax') #gives percentages for each number in third layer
    ])
        
#Compile the model/properties of model(giving extra features/personalizing)(gather raw materials to build)
model.compile(optimizer='adam', #one of the image optimizers(tries to get a smaller difference(loss))
              loss='sparse_categorical_crossentropy', #Difference between input and output
              metrics= ['accuracy']) #A form of unit

#Train the model(Building the house)
model.fit(train_images,train_labels,epochs=15)

#Test the model(Living in the house and checking if everything is fine)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)  #accuracy of test

model.save('StephenCurryKevinDurantImgClassifier.h5')



