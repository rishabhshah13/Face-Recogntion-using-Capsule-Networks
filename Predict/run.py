from Model import *
import numpy as np
import cv2
import os
import h5py
import dlib
from imutils import face_utils
from keras.models import load_model
import sys

_,model = CapsNetLight((28,28,1), 5, 3)
model.load_weights('trained_model.h5')
model.summary()
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
font = cv2.FONT_HERSHEY_SIMPLEX
def start():
    cap = cv2.VideoCapture(0)
    print('here')
    ret = True

    clip = []
    while ret:
        #read frame by frame
        ret, frame = cap.read()
        nframe = frame
        faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #flags=cv2.CV_HAAR_SCALE_IMAGE
        )
        # resize to the test size
        #    tmp_ = center_crop(cv2.resize(frame, (171, 128)))

        #seems normalize color
        #   tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        #print(faces)
        try:
            (x,y,w,h) = faces[0]
        except:
            continue
        frame = frame[y:y+h,x:x+w]
        frame = cv2.resize(frame, (28,28))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('result small' , frame) 
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break
        
        #gray = gray[np.newaxis,:,:,np.newaxis]
        gray = gray.reshape(-1, 28, 28, 1).astype('float32') / 255.
        print(gray.shape)
        prediction = model.predict(gray)[0]
        print(prediction)

        
        print("\n\n\n\n")
        print("----------------------------------------------")
        labels = ['Colin Powell' ,'Donald Rumsfeld' ,'George W Bush' ,'Tony Blair','Rishabh']
        prediction = prediction.tolist()
        i = 0
        for label in labels:
            #print( label + "==>" + str(prediction[0][i]) )
            i = i + 1
        listv = prediction[0]
        n = listv.index(max(listv))
        print("\n")
        print("----------------------------------------------")
        print( "Highest Probability: " + labels[n] + "==>" + str(prediction[0][n]) )
        print("----------------------------------------------")
        print("\n")
        for (x, y, w, h) in faces:
            try:
                cv2.rectangle(nframe, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(nframe, str(labels[n]), (x+5,y-5), font, 1, (255,255,255), 2)
                #cv2.putText(nframe, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1) 
            except:
                la = 2 
        prediction = np.argmax(model.predict(gray)[0], 1)
        print(prediction)
        cv2.imshow('result', nframe)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

        


        
def test():
    #image = cv2.imread('/home/rishabh/Desktop/face/OpenCV-Face-Recognition-master WORKING/FacialRecognition/dataset/User.1.9.jpg')
    image = cv2.imread('/home/rishabh/Desktop/George_W_Bush_0001.jpg')
    
    cv2.imshow('lol',image)
    cv2.waitKey(0)
    frame = image
    
    faces = faceCascade.detectMultiScale(
    frame,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),)

    try:
        (x,y,w,h) = faces[0]
    except:
        l = 1
    #frame = frame[y:y+h,x:x+w]
    frame = cv2.resize(frame, (47,62))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('lol',gray)
    cv2.waitKey(0)
    #frame = np.array(frame).astype(np.float32)        
    #frame = frame[np.newaxis,:,:]
    gray = gray.reshape(-1, 62, 47, 1).astype('float32') / 255.
    print(gray.shape)
    #gray = gray[np.newaxis,:,:]
    prediction = model.predict(gray)[0]
    print(prediction)
    prediction = np.argmax(model.predict(gray)[0], 1)
    #prediction = model.predict(gray)[0]
    #a = np.sum(np.argmax(prediction, 1))
    #print(a)
    print(prediction)
    
#test()
start()
