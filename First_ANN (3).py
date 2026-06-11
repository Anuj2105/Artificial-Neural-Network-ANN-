#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import style
style.use('ggplot')

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings('ignore')


# In[2]:


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


type(x_train)


# In[4]:


print('Training data shape    : ', x_train.shape)
print('Training labels shape  : ', y_train.shape)
print('Testing data shape     : ', x_test.shape)
print('Testing labels shape   : ', y_test.shape)


# In[5]:


plt.imshow(x_train[1], cmap = 'gray')
plt.axis('off')


# In[6]:


np.set_printoptions(linewidth = 300)
print(x_train[1])


# In[7]:


plt.figure(figsize = (20,10))
for i in range(200):
  plt.subplot(10,20, i+1)
  plt.imshow(x_train[i], cmap = 'gray')
  plt.axis('off')


# In[8]:


x_train = x_train.reshape(60000, 28,28,1)
x_test  = x_test.reshape(10000, 28,28,1)


# In[9]:


# Encoding the labels
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes = 10)


# In[10]:


print('Training data shape    : ', x_train.shape)
print('Training labels shape  : ', y_train.shape)
print('Testing data shape     : ', x_test.shape)
print('Testing labels shape   : ', y_test.shape)


# In[11]:


train_datagen = ImageDataGenerator(
    rescale = 1./255,
    # rotation_range = 0.1,
    # zoom_range = 0.1,
    # horizontal_flip = True,
    # vertical_flip = True,
    # shear_range = 0.1,
    width_shift_range = 0.1,
    height_shift_range = 0.1)

val_datagen = ImageDataGenerator(rescale = 1./255)


# In[12]:


train_datagen.fit(x_train)
val_datagen.fit(x_test)

train_generator = train_datagen.flow(x_train, y_train, batch_size = 1000)
val_generator = val_datagen.flow(x_test, y_test, batch_size = 1000)


# In[13]:


# Stop the training when accuracy is more than threshold
from keras.callbacks import Callback

threshold_accuarcy = 0.985

class MyCallback(Callback):
  def on_epoch_end(self, epoch, logs = {}):
    if logs['val_accuracy'] > threshold_accuarcy:
      print('\nReached {}% accuracy so cancelling training!'.format(threshold_accuarcy*100))
      self.model.stop_training = True

my_callback = MyCallback()


# In[14]:


# Learning rate reduction
from keras.callbacks import ReduceLROnPlateau
lr_reduction = ReduceLROnPlateau(monitor = 'accuracy',
                                 patience = 2,
                                 factor = 0.5,
                                 verbose = 1)


# # Steps to create and train a neural network
# 1. Create the architecture
# 2. Compile the model
# 3. Fit the model

# In[15]:


# Creating the architecture
model = tf.keras.Sequential()
model.add(Flatten(input_shape = (28,28,1)))
model.add(Dense(200, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()


# In[16]:


# Compile the model
model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(),
              metrics = ['accuracy'])


# In[17]:


# Fit the model
history = model.fit(
    train_generator,
    validation_data = (val_generator),
    epochs = 100,
    verbose = 1,
    callbacks = [my_callback, lr_reduction])


# In[18]:


type(history.history)


# In[19]:


history.history.keys()


# In[20]:


acc_train = history.history['accuracy']
acc_val = history.history['val_accuracy']

loss_train = history.history['loss']
loss_val = history.history['val_loss']


# In[22]:


x = range(1,len(acc_train) + 1)

plt.figure(figsize = (24,8), dpi = 300)
plt.subplot(1,2,1)
sns.lineplot(x = x, y = acc_train, label = 'Training Accuracy')
sns.lineplot(x = x, y = acc_val, label = 'Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
sns.lineplot(x = x, y = loss_train, label = 'Training Loss')
sns.lineplot(x = x, y = loss_val, label = 'Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.show()


# In[23]:


# Making predictions
model.predict(x_train[0:1])


# In[24]:


np.argmax(model.predict(x_train[0:1]))


# In[25]:


model.predict(x_train[0:10])


# In[ ]:


np.argmax(model.predict(x_train[0:10]), axis = 1)


# In[ ]:


np.argmax(y_train[0:10], axis = 1)


# In[ ]:




