import os
import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from keras.applications.xception import Xception,preprocess_input
import tensorflow as tf
from keras.preprocessing import image
from keras.layers import Input
from keras.backend import reshape
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# images_dir = '../input/h-and-m-personalized-fashion-recommendations/images'
images_dir = '/kaggle/input/random-images'

def getImagePaths(path):
    """
    Function to Combine Directory Path with individual Image Paths

    parameters: path(string) - Path of directory
    returns: image_names(string) - Full Image Path
    """
    image_names = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            fullpath = os.path.join(dirname, filename)
            image_names.append(fullpath)
    return image_names

def preprocess_img(img_path):
    try:
        dsize = (225,225)
        new_image = cv2.imread(img_path)
        new_image = cv2.resize(new_image,dsize,interpolation=cv2.INTER_NEAREST)
        new_image = np.expand_dims(new_image,axis=0)
        new_image = preprocess_input(new_image)
        return new_image
    except Exception as e:
        return None

def load_data():
    output = []
    output = getImagePaths(images_dir)[:10000]
    return output

def model():
    model = Xception(weights='imagenet',include_top=False)
    for layer in model.layers:
        layer.trainable=False
#         model.summary()
    return model

def feature_extraction(image_data,model):
    features = model.predict(image_data)
    features = np.array(features)
    features = features.flatten()
    return features

features = []
output = load_data()
main_model = model()

#Limiting the data for training
for image_train in output[:999]:
    new_img = preprocess_img(image_train)
    if new_img is not None:
        features.append(feature_extraction(new_img, main_model))
feature_vec = np.array(features)

def result_vector_cosine(model,feature_vector,new_img):
    new_feature = model.predict(new_img)
    new_feature = np.array(new_feature)
    new_feature = new_feature.flatten()
    N_result = 12
    nbrs = NearestNeighbors(n_neighbors=N_result, metric="cosine").fit(feature_vector)
    distances, indices = nbrs.kneighbors([new_feature])
    return(indices)

def input_show(path):
    img = plt.imread(path)
    plt.title("Query Image")
    plt.imshow(img)

def show_result(data,result):
    fig = plt.figure(figsize=(12,8))
    for i in range(0,12):
        index_result=result[0][i]
        plt.subplot(3,4,i+1)
        plt.imshow(cv2.imread(data[index_result]))
    plt.show()

path_query = '/kaggle/input/random-images/dataset-image/lion/lion_11.jpg'
result = result_vector_cosine(main_model, feature_vec, preprocess_img(path_query))
input_show(path_query)
show_result(output,result)
