import numpy as np
import seaborn as sns
import matplotlib as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

tf.__version__
'''
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D 
from keras.layers import Flatten 
from keras.layers import Dense
'''

# Generating images for the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   rotation_range=15)

# Generating images for the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creating the Training set
training_set = train_datagen.flow_from_directory(r'C:\Users\dogra\Desktop\sirdard\Project\train',
                                                 target_size = (48,48),
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
                                                 color_mode='grayscale',
                                                 shuffle=True)

# Creating the Test set
test_set = train_datagen.flow_from_directory(r'C:\Users\dogra\Desktop\sirdard\Project\test',
                                             target_size = (48, 48),
                                             batch_size = 32,
                                             class_mode = 'categorical',
                                             color_mode='grayscale',
                                             shuffle=False)

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[48, 48, 1]))

#cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, padding='same'))
cnn.add(tf.keras.layers.BatchNormalization())

#SECOND LATER---------------
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))

# third layer------------------
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, padding='same'))

# fourth layer------------------
cnn.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, padding='same'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.Dropout(0.25))

#Flattening
cnn.add(tf.keras.layers.Flatten())

#  Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())

cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))

cnn.add(tf.keras.layers.Dense(units=7, activation='softmax'))


# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

cnn.summary() 

# training_set.n to get size 

#TO SAVE WEIGHTS AS CHECKPOINT
from keras.callbacks import CSVLogger,ModelCheckpoint, ReduceLROnPlateau

checkpoint = ModelCheckpoint("weights.h5", monitor='accuracy', verbose=1, 
                             save_best_only=True, save_weights_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, min_lr=0.0001, mode='auto')

callbacks = [checkpoint, reduce_lr]
#steps_per_epoch = training_set.n//training_set.batch_size
#validation_steps = test_set.n//test_set.batch_size


history=cnn.fit_generator(training_set,
                          steps_per_epoch = 897,
                          epochs = 20,
                          validation_data = test_set,
                          validation_steps = 224,
                          callbacks=callbacks
                          )


#pip install h5py 
#cnn.save_weights('weights.h5',overwrite='True')
# cnn.save() to save whole model
model_json = cnn.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# for graph
epochs=20
from matplotlib import pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



# SAVE JSON FILE 
''' FOR MORE DETAILS SEE:--
https://machinelearningmastery.com/save-load-keras-deep-learning-models/

cnn.save_weights('weights.h5',overwrite='True') #pip install h5py 
then load this file this was also working
'''

