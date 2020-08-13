import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tensorflow.keras import regularizers
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 45,
                                   width_shift_range = .15,
                                   height_shift_range = .15,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (150, 150),
                                                 batch_size = 64,
                                                 shuffle = True,
                                                class_mode = 'categorical')
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('test',
                                            target_size = (150, 150),
                                            batch_size = 64,
                                            class_mode = 'categorical')
##Architecture of CNN
cnn = tf.keras.Sequential()
#Convolutional layers and Max pooling
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[150,150, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))         
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
#Flattening 
cnn.add(tf.keras.layers.Flatten())
# 3 hidden layers 
cnn.add(tf.keras.layers.Dense(units=128, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
cnn.add(tf.keras.layers.Dense(units=128, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
cnn.add(tf.keras.layers.Dense(units=128, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
cnn.add(tf.keras.layers.Dropout(0.3))
# output layer
cnn.add(tf.keras.layers.Dense(units=4, activation='softmax'))
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
cnn.fit(x = training_set, validation_data = test_set, epochs = 60)

##Save the model
cnn.save("mymodel.h5")
