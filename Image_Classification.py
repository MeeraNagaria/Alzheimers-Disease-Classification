# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 20:09:58 2021

@author: meera
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from PIL import Image
import glob
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import pathlib

dataset_dir="C:/Alzheimer_s Dataset/train/"
len(os.listdir(dataset_dir))
images=glob.glob("*.jpg")
#data_dir = tf.keras.utils.get_file('train', origin=dataset_dir)

data_dir = pathlib.Path(dataset_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

#mildDemented = list(data_dir.glob('mildDem*.jpg'))
#PIL.Image.open(str(mildDemented[0]))
#
#ModerateDemented = list(data_dir.glob('moderateDem*.jpg'))
#PIL.Image.open(str(ModerateDemented[0]))
#
#NonDemented = list(data_dir.glob('nonDem*.jpg'))
#PIL.Image.open(str(NonDemented[0]))
#
#VeryMildDem = list(data_dir.glob('verymildDem*.jpg'))
#PIL.Image.open(str(VeryMildDem[0]))

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,labels='inferred',label_mode='int',
  class_names=None, color_mode='rgb',  image_size=(img_height, img_width),
  batch_size=batch_size, shuffle=True, seed=123, validation_split=0.2, subset="training",
    interpolation='bilinear', follow_links=False, smart_resize=False)
  
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
   image_size=(img_height, img_width),
  batch_size=batch_size,interpolation='bilinear', follow_links=False, smart_resize=False)


class_names = train_ds.class_names
print(class_names)  

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break    

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = 4

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
    

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])    

model.summary()

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#data_augmentation = keras.Sequential(
#      [
#        layers.experimental.preprocessing.RandomFlip("horizontal", 
#                                                     input_shape=(img_height, 
#                                                                  img_width,
#                                                                  3)),
#        layers.experimental.preprocessing.RandomRotation(0.1),
#        layers.experimental.preprocessing.RandomZoom(0.1),
#      ]
#    )
#  
#plt.figure(figsize=(10, 10))
#for images, _ in train_ds.take(1):
#  for i in range(9):
#    augmented_images = data_augmentation(images)
#    ax = plt.subplot(3, 3, i + 1)
#    plt.imshow(augmented_images[0].numpy().astype("uint8"))
#    plt.axis("off")
#    
#    
#model = Sequential([
#  data_augmentation,
#  layers.experimental.preprocessing.Rescaling(1./255),
#  layers.Conv2D(16, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Conv2D(32, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Conv2D(64, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Dropout(0.2),
#  layers.Flatten(),
#  layers.Dense(128, activation='relu'),
#  layers.Dense(num_classes)
#])    
#      
#model.compile(optimizer='adam',
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#              metrics=['accuracy'])      
#model.summary()
#
#acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
#
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#
#epochs_range = range(epochs)
#
#plt.figure(figsize=(8, 8))
#plt.subplot(1, 2, 1)
#plt.plot(epochs_range, acc, label='Training Accuracy')
#plt.plot(epochs_range, val_acc, label='Validation Accuracy')
#plt.legend(loc='lower right')
#plt.title('Training and Validation Accuracy')
#
#plt.subplot(1, 2, 2)
#plt.plot(epochs_range, loss, label='Training Loss')
#plt.plot(epochs_range, val_loss, label='Validation Loss')
#plt.legend(loc='upper right')
#plt.title('Training and Validation Loss')
#plt.show() 

# PREDICTION a new data point
#text="C:/Alzheimer_s Dataset/test/testing/29 (2).jpg"
def predImage(text):
    dataset_dir_pred=text
    
    img = keras.preprocessing.image.load_img(
        dataset_dir_pred, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    out = (
        'This image most likely belongs to {} with a {:.2f} percent confidence.'
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    return out
    
#predImage("C:/Alzheimer_s Dataset/test/testing/29 (2).jpg")
