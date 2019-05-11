import os
import pickle
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import numpy as np
import random
import cv2
from keras.utils import to_categorical

def downsample_image(img):
    img = cv2.resize(img, (28,28))
    return np.array(img)

def get_face_data(min_faces_per_person):
    people = fetch_lfw_people(color=False, min_faces_per_person = min_faces_per_person)
    X_faces = people.images
    Y_faces = people.target

    X_faces = np.array([downsample_image(ab) for ab in X_faces])
    X_train, X_test, y_train, y_test = train_test_split(X_faces, Y_faces,
                                                        test_size=0.2)
    return X_train, y_train, X_test, y_test
    
    
def load_data(min_faces_per_person):


    x_train, y_train, x_test, y_test = get_face_data(min_faces_per_person)

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)



