# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:39:58 2019

@author: HARSH
"""

from keras.models import Sequential 
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense  

clasffier = Sequential() 

clasffier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

clasffier.add(MaxPooling2D(pool_size = (2, 2)))

clasffier.add(Flatten())

clasffier.add(Dense(output_dim = 128, activation = 'relu'))
clasffier.add(Dense(output_dim = 1, activation = 'sigmoid'))

clasffier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

clasffier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=20,
        validation_data=test_set,
        nb_val_samples=2000)