# import libraries
import numpy as np
import pandas as pd
import cv2
import os
import random as rn
from random import shuffle
from zipfile import ZipFile
from PIL import Image
import warnings

import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
style.use('fivethirtyeight')
sns.set(style='whitegrid', color_codes=True)

# set random seeds
np.random.seed(0)
rn.seed(0)

labels = ['PNEUMONIA', 'NORMAL']
img_size = 150
def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

# import data
train = get_data(r'C:\PycharmProjects\chest-x-ray-pneumonia\archive\chest_xray\chest_xray\train')
val = get_data(r'C:\PycharmProjects\chest-x-ray-pneumonia\archive\chest_xray\chest_xray\val')
test = get_data(r'C:\PycharmProjects\chest-x-ray-pneumonia\archive\chest_xray\chest_xray\test')
print('the number of images in training set:', len(train))
print('the number of images in validation set:', len(val))
print('the number of images in test set:', len(test))

# combine train and val and divide them again with 8:2 ratio
temp = np.concatenate([train, val], axis=0)
len(temp)
train, val = train_test_split(temp, test_size=0.2, random_state=0)
print('the number of images in training set:', len(train))
print('the number of images in validation set:', len(val))

# compare the number of cases and non-cases
l = []
for i in train:
    if(i[1] == 0):
        l.append("Pneumonia")
    else:
        l.append("Normal")
sns.set_style('darkgrid')
sns.countplot(l)

# visualize images
plt.figure(figsize = (5,5))
plt.imshow(train[0][0], cmap='gray')
plt.title(labels[train[0][1]])

plt.figure(figsize = (5,5))
plt.imshow(train[-1][0], cmap='gray')
plt.title(labels[train[-1][1]])

# separate features and labels
x_train = []
y_train = []

x_val = []
y_val = []

x_test = []
y_test = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

# perform grayscale normalization
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255

# resize data for deep learning
x_train = x_train.reshape(-1, img_size, img_size, 3)
y_train = np.array(y_train)

x_val = x_val.reshape(-1, img_size, img_size, 3)
y_val = np.array(y_val)

x_test = x_test.reshape(-1, img_size, img_size, 3)
y_test = np.array(y_test)

# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2,   # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

# specify the base model (transfer learning) and build from there
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(150,150,3), pooling='avg')
base_model.trainable = False # freeze the pre-trained weights from VGG16
## to unfreeze some layers
# for i in range(len(base_model.layers)):
#     print(i, base_model.layers[i])
#
# for layer in base_model.layers[15:]:
#     layer.trainable = True
# for layer in base_model.layers[0:15]:
#     layer.trainable = False
base_model.summary()

# or build a model form scratch
# model = Sequential()
# model.add(base_model)
# model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same', activation='relu', input_shape=(150,150,3)))
# model.add(BatchNormalization())
# model.add(MaxPool2D((2,2), strides=2, padding = 'same'))
# model.add(Conv2D(64, (3,3), strides=1, padding='same', activation='relu'))
# model.add(Dropout(0.1))
# model.add(BatchNormalization())
# model.add(MaxPool2D((2,2), strides=2, padding='same'))
# model.add(Conv2D(64, (3,3), strides=1, padding='same', activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPool2D((2,2), strides=2, padding='same'))
# model.add(Conv2D(128, (3,3), strides=1, padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
# model.add(MaxPool2D((2,2), strides=2, padding='same'))
# model.add(Conv2D(256, (3,3), strides=1, padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
# model.add(MaxPool2D((2,2), strides=2, padding='same'))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))
# model.summary()

# fit the model
model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.3,
                                            min_lr=0.000001)
history = model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=1,
                    validation_data=datagen.flow(x_val, y_val), callbacks=[learning_rate_reduction])

# evaluate the model
print("Loss of the model is - " , model.evaluate(x_test,y_test)[0])
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")


epochs = [i for i in range(12)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(20,10)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
ax[1].set_title('Training & Validation Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()

# predict on the test set
predictions = model.predict_classes(x_test)
predictions = predictions.reshape(1,-1)[0]
predictions[:15]

# model performance on the test set
print(classification_report(y_test, predictions, target_names = ['Pneumonia (Class 0)','Normal (Class 1)']))

cm = confusion_matrix(y_test,predictions) # confusion matrix
cm

cm = pd.DataFrame(cm, index=['0','1'], columns=['0','1'])
plt.figure(figsize=(10,10))
sns.heatmap(cm, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='',
            xticklabels=labels, yticklabels=labels)

correct = np.nonzero(predictions == y_test)[0]
incorrect = np.nonzero(predictions != y_test)[0]