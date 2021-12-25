# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:40:07 2020

@author: liuti
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:52:00 2020

@author: liuti
"""
#part1-----------------------------building CNN
#importing the libraries
from keras.models import Sequential #initialize the CNN
from keras.layers import Conv2D #add the convolutional layers, 2D because pictures are 2D
from keras.layers import MaxPooling2D #add the pooling layers
from keras.layers import Flatten #convert all the pooled feature maps into a large feature vector, then become the input of the fully connected layer
from keras.layers import Dense #used to add the fully connected layer to the ANN
from keras.preprocessing.image import ImageDataGenerator

#step 1: initializing the CNN
classifier=Sequential()
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))

#step 2: pooling
classifier.add(MaxPooling2D(pool_size=(2,2))) # (2,2) for most of the time

#add second convolutional layer
classifier.add(Conv2D(32,(3,3),activation='relu')) #no need of the input_shape
classifier.add(MaxPooling2D(pool_size=(2,2))) # (2,2) for most of the time

#step 3: flattening
classifier.add(Flatten())

#step 4: full connection
classifier.add(Dense(units=128,activation='relu')) 
classifier.add(Dense(units=1,activation='sigmoid')) 

#compile the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#---------------------picture preprocessing to reduce overfitting (use image augmentation to enrich the pictures by tilting or roatating the training pictures)
train_datagen = ImageDataGenerator(
        rescale=1./255, #scale the pixel value to the range [0,1]
        shear_range=0.2, #random transformation
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255) #only rescale here

trainning_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary') #binary outcome

classifier.fit_generator(
        trainning_set,
        steps_per_epoch=8000, #8000 training pictures
        epochs=25,
        validation_data=test_set,
        validation_steps=2000) #2000 test pictures





