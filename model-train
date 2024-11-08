import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2

# noinspection PyUnresolvedReferences
from keras.utils import to_categorical
# noinspection PyUnresolvedReferences
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout
# noinspection PyUnresolvedReferences
from keras.models import Sequential,load_model
# noinspection PyUnresolvedReferences
from sklearn.model_selection import train_test_split

np.random.seed(1)

train_images = []
train_labels = []
shape = (200,200)
train_path = '{Training_Dataset_Path}'

for filename in os.listdir(train_path):
    if filename.split('.')[1] == 'png':
        img = cv2.imread(os.path.join(train_path, filename))
        train_labels.append(filename.split('_')[0])
        img = cv2.resize(img, shape)
        train_images.append(img)

train_labels = pd.get_dummies(train_labels).values
train_images = np.array(train_images)
x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, random_state=1)

test_images = []
test_labels = []
shape = (200, 200)
test_path = '{Test_Dataset_Path}'

for filename in os.listdir(test_path):
    if filename.split('.')[1] == 'png':
        img = cv2.imread(os.path.join(test_path, filename))
        test_labels.append(filename.split('_')[0])
        img = cv2.resize(img, shape)
        test_images.append(img)

test_images = np.array(test_images)

# Creating a Sequential model
model = Sequential()
model.add(Conv2D(kernel_size=(3, 3), filters=32, activation='tanh', input_shape=(200, 200, 3,)))
model.add(Conv2D(filters=30, kernel_size=(3, 3), activation='tanh'))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(filters=30, kernel_size=(3, 3), activation='tanh'))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(filters=30, kernel_size=(3, 3), activation='tanh'))

model.add(Flatten())

model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='relu'))
# PROBABLY the output=final layer, which commonly uses softmax
model.add(Dense(2, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    metrics=['acc'],
    optimizer='adam'
)

# Model Summary
model.summary()

# Training the model
history = model.fit(x_train,y_train,epochs=50,batch_size=50,validation_data=(x_val,y_val))

# Saving the model
model.save(os.path.join('models','diagram_classification_model.keras'))

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
