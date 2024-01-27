import streamlit as st 
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.applications.resnet50 import preprocess_input
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
neighbors=NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
file_path=r"artifacts\\training\\imagespath.pkl"
with open(file_path, "rb") as path :
    image_paths = pickle.load(path)
all_features=r"artifacts\\training\\extracted_feature.pkl"
with open(all_features,'rb') as features:
    images_features=pickle.load(features)
model=load_model(r"artifacts\\prepare_base_model\\base_model_updated.h5")
model.compile(loss='mse',)
def feature_extraction(img,model):
    
    img=img.resize((224,224))
    img_array = img.convert('RGB')
    img_array = np.array(img_array)
    img=np.expand_dims(img_array,axis=0)
    preprocessed_img=preprocess_input(img)
    result=model.predict(preprocessed_img).flatten()
    normalized_output=result/norm(result)
    return normalized_output
def recommender(image_features,img_feature):
    neighbors.fit(image_features)
    distance,indices=neighbors.kneighbors([img_feature])
    return indices

st.title("Fashion Recommender System")
file_upload=st.file_uploader('Choose image',type=['jpg','png'])

if file_upload is None:
    print('Please upload an image')
else:
    img=Image.open(file_upload)
    st.image(img,width=300)
    img_feature=feature_extraction(img,model)
    indices=recommender(images_features,img_feature)
    image1,image2,image3,image4,image5=st.columns(5)
    with image1:
        st.image(Image.open(image_paths[indices[0][0]]),width=100)
    with image2:    
        st.image(Image.open(image_paths[indices[0][1]]),width=100)
    with image3:
        st.image(Image.open(image_paths[indices[0][2]]),width=100)
    with image4:
        st.image(Image.open(image_paths[indices[0][3]]),width=100)
    with image5:
        st.image(Image.open(image_paths[indices[0][4]]),width=100)


    
