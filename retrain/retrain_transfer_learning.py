from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
top_model_path = '../model_and_app/model.h5'
top_model_weights_path = '../model_and_app/weights.h5'

new_model_path = 'transfer_learning_model.h5'
new_model_weights_path = 'transfer_learning_new_weights.h5'


train_dir = 'data/train'
validation_dir = 'data/validation'


train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

batch_size = 128
epochs = 20
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Generator for our training data
image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    ) 
image_gen_val = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )  # Generator for our validation data

train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

model = tensorflow.keras.models.load_model(top_model_path)


# for i,layer in enumerate(model.layers):
#   print(i,layer.name)

for layer in model.layers[:7]:
    layer.trainable=False
for layer in model.layers[7:]:
    layer.trainable=True




op = Adam(lr=0.0003)
model.compile(optimizer=op,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

model.summary()
# model.save(new_model_path)
# print(history.history.keys())
# 'acc', 'loss', 'val_acc', 'val_loss'
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

fig = plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.ylim(0.5, 0.85)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.ylim(0.35, 0.75)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
fig.savefig('result_exp3/acc_loss_basic.png')


np.set_printoptions(precision=4) 
np.savetxt("result_exp3/exp3_acc_history.txt", np.array(acc), delimiter=",")
np.savetxt("result_exp3/exp3_val_acc_history.txt", np.array(val_acc), delimiter=",")
np.savetxt("result_exp3/exp3_loss_history.txt", np.array(loss), delimiter=",")
np.savetxt("result_exp3/exp3_val_loss_history.txt", np.array(val_loss), delimiter=",")