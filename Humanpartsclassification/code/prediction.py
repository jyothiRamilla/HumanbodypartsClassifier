import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
#from keras.models import load_model

model = tf.keras.load_model("model.h5")
classes = ["Ears","Eyes","Hands","Legs"]
test_image = image.load_img('Pathtotestimage', target_size = (150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
print(result)
print(classes[np.argmax(np.array(result))])
