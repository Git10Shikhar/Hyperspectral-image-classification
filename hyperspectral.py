#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 22:12:17 2020

@author: vivek
"""

import keras
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, BatchNormalization
from keras.layers import Dropout, Input
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

from operator import truediv

from plotly.offline import init_notebook_mode

import numpy as np

import matplotlib.pyplot as plt

import os

import spectral

from scipy.io import loadmat



def LoadData():
    
    data_path = os.path.join(os.getcwd(),'data')
    data = loadmat(os.path.join('Indian_pines_corrected.mat'))['indian_pines_corrected']
    labels = loadmat(os.path.join('Indian_pines_gt.mat'))['indian_pines_gt']
    return data, labels




    
def TrainTestSplit(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=345)
    return X_train, X_test, y_train, y_test





def applyPCA(X, numComponents=30):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca





def PWZ(X,margin=2):     #padding with zeros
    newX=np.zeros((X.shape[0]+ 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    X_offset=margin
    y_offset=margin
    newX[X_offset:X.shape[0] + X_offset , y_offset:X.shape[0] + y_offset:]=X
    return newX





def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = PWZ(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

    
    


X, y = LoadData()

X,pca = applyPCA(X)

X, y = createImageCubes(X, y, windowSize=25)

Xtrain, Xtest, ytrain, ytest = TrainTestSplit(X, y)

print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

Xtrain = Xtrain.reshape(-1, 25, 25, 30 ,1)
ytrain = np_utils.to_categorical(ytrain)





# NETWORK ---------------->

input_layer = Input((25, 25, 30, 1))

conv_layer1 = Conv3D(8, kernel_size=(3, 3, 7), activation='relu')(input_layer)

conv_layer2 = Conv3D(16, kernel_size=(3, 3, 5), activation='relu')(conv_layer1)

conv_layer3 = Conv3D(32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)

print(conv_layer3._keras_shape)

conv3d_shape = conv_layer3._keras_shape
conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)

conv_layer4 = Conv2D(64,kernel_size=(3,3),activation='relu')(conv_layer3)

flatten_layer = Flatten()(conv_layer4)

dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)

dense_layer1 = Dropout(0.4)(dense_layer1)

dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)

dense_layer2 = Dropout(0.4)(dense_layer2)

output_layer = Dense(units=16, activation='softmax')(dense_layer2)





model = Model(inputs=input_layer, outputs=output_layer)


model.summary()


model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

model.fit(x=Xtrain, y=ytrain, batch_size=256, epochs=100)






