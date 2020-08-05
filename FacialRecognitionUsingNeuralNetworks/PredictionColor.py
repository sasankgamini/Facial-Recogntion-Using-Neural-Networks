import numpy as np
import cv2
import keras
from keras.models import load_model

model=load_model('SCKDColorImgClassifier.h5')

facecascade=cv2.CascadeClassifier('haarcascade_frontalface.xml')
image = cv2.imread('Both2.jpg')
face=facecascade.detectMultiScale(image,1.1,5)
for feature in face:
    print('worked')
    cv2.rectangle(image,
                  (feature[0],feature[1]),
                  (feature[0]+feature[2],
                   feature[1]+feature[3]),(0,255,0),2)
    ROI = image[feature[1]:feature[1]+feature[3],
                    feature[0]:feature[0]+feature[2]]
    ROI=cv2.resize(ROI,(50,50))
    dilatedimg=cv2.dilate(ROI,(3,3))
    dilatedlist=[dilatedimg] #shape needs to be array so made it a list
    dilatedarray=np.array(dilatedlist) #shape needs to be (1,50,50,3) so we make it an array
    print(dilatedarray.shape)
    dilatedarray=dilatedarray/255 #range only from 0 to 1 so more accurate
    
    predictions=model.predict(dilatedarray)

    players=['Kevin Durant','Stephen Curry']
    print(predictions)
    maximum=np.argmax(predictions[0])
    print(players[maximum])

    cv2.putText(image, str(players[maximum]), (feature[0]-100,feature[1]+50),cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2) 


cv2.imshow('face',image)
cv2.waitKey()
cv2.destroyAllWindows()
