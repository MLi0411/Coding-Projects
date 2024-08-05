# Margaret Li
# 3/9/22
# EPS ATiCS
# This program loads a dataset from Kaggle that contains MRI
# images of brains with and without tumors. This program 
# visualizes the data, trains a cnn to classify tumors from the
# images, makes predictions, and shows these predictions 
# in a confusion matrix visualization

# for loading data
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import random
import os
import tensorflow as tf

# for CNN
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

# for visualizations
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 



### Loading data ###
# data description:
# this dataset contains MRI images of brains with pituitary, meningioma, and 
# glioma tumors or with no tumors. There are 5712 images in train set and 
# 1311 images in test set. Images are 224 * 224 and all in grayscale. 

# source of data: https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset
# source of code for loading data: https://www.kaggle.com/harshwalia/brain-tumor-classification-transfer-learning
# used code above with minor changes from line 38 - 117

# loading the train set from local folder
train_dir = "C:\\Users\\jli\\OneDrive - Eastside Preparatory School\\Adv Topics\\brain tumor dataset\\Training\\"
train_paths = []
for label in os.listdir(train_dir):
    for file in os.listdir(train_dir+label):
        train_paths.append(train_dir+label+'\\'+file)
random.shuffle(train_paths)

# loading the test set from local folder
test_dir = "C:\\Users\\jli\\OneDrive - Eastside Preparatory School\\Adv Topics\\brain tumor dataset\\Testing\\"
test_paths = []
for label in os.listdir(test_dir):
    for file in os.listdir(test_dir+label):
        test_paths.append(test_dir+label+'\\'+file)
random.shuffle(test_paths)

# create lists for train data and test data
# model will be trained to predict things based on
# train data and train labels
X_train = []
y_train = []

# model will make predictions for test data
# and results can be validated with test labels
X_test = []
y_test = []


# update train set
# loop over the image paths
for imagePath in train_paths:
    # get the label
    label = imagePath.split(os.path.sep)[-2]

    # get the input image (224x224)
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = np.array(image)/255.0

    # update train set with images for X and labels for y
    X_train.append(image)
    y_train.append(label)

# changing X_train list of images to numpy list
# this is important for processing in CNN
X_train = tf.stack(X_train)

# for all train set labels, change string names 
# to integers. These integers will be important in CNN
for i in range(0, 5712):
    if y_train[i] == "glioma":
        y_train[i] = 0
    if y_train[i] == "meningioma":
        y_train[i] = 1
    if y_train[i] == "notumor":
        y_train[i] = 2
    if y_train[i] == "pituitary":
        y_train[i] = 3


# update test set
# loop over the image paths
for imagePath in test_paths:
    # get the label
    label = imagePath.split(os.path.sep)[-2]

    # get the input image (224x224)
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = np.array(image)/255.0

    # update train set with images for X and labels for y
    X_test.append(image)
    y_test.append(label)

# changing X_train list of images to numpy list
# this is important for processing in CNN
X_test = tf.stack(X_test)

# for all train set labels, change string names 
# to integers. These integers will be important in CNN
for i in range(0, 1311):
    if y_test[i] == "glioma":
        y_test[i] = 0
    if y_test[i] == "meningioma":
        y_test[i] = 1
    if y_test[i] == "notumor":
        y_test[i] = 2
    if y_test[i] == "pituitary":
        y_test[i] = 3



### preprocess visualization ###
# 9 examples of MRI images in the data set with
# their respected labels would extracted and
# visualized
# using seaborn to plot
sns.set(font_scale=2)
# select 9 data points from dataset
index = np.random.choice(np.arange(len(X_train)), 9, replace=False)
# set subplots of data
figure, axes = plt.subplots(nrows=3, ncols=3, figsize=(16,9))

# organize the images and labels for the visualization
for item in zip(axes.ravel(), index):
    axes, idx = item
    image = X_train[idx]
    target = y_train[idx]
    axes.imshow(image, cmap=plt.cm.gray_r) # show image
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target) # set title to target (the correct answers)
# better set up. less white space and more plots
plt.tight_layout()

plt.show()



### processing data ###
# turns each label into one-hot vector
# an index in the vector would represent the label 
# instead of an integer
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# model fitting an prediction #
# CNN model would be trained with data from train set
# it would then be fitted with data from test set and
# it would make predictions the same way. Predictions
# can be compared to labels from y_test
cnn = Sequential()

# add layers
# detect features with convolutional layer
cnn.add(Conv2D(filters=64, # number of times looking at an image kernel by kernel
               kernel_size=(3, 3), # kernel shape is 3*3
               activation='relu', # activation function
               input_shape=(224, 224, 3))) # specify dimensions of images

# reduce overfitting by pooling
cnn.add(MaxPooling2D(pool_size=(2, 2)))

# expands with convolutional layer with even more filters
cnn.add(Conv2D(filters=128, 
               kernel_size=(3, 3),
               activation='relu'))

# pools again
cnn.add(MaxPooling2D(pool_size=(2, 2)))

# flatten into 1D array
cnn.add(Flatten())


# dense layers 
# since we already have gathered attributes, do the prediction now
cnn.add(Dense(units=128, activation='relu'))
# reduce units until fits number of categories
cnn.add(Dense(units=4, activation='softmax'))

# print CNN summary of layers
print(cnn.summary())
print("")


# optimizer
# Keeps the same structure but picking the inaccuracies and correcting
# put the layers together into a network
cnn.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# per epoch, put data into neural networks to tune
# print results after each epoch
cnn.fit(X_train, y_train, epochs=3, batch_size=110, validation_split=0.1)

# evaluate model on test sets and make predictions
loss, accuracy = cnn.evaluate(X_test, y_test)
prediction = cnn.predict(X_test)

# print stats of prediction
print("")
print("Loss:")
print(loss)
print("Accuracy:")
print(accuracy)

# change y_test from one-hot vectors to
# integers that correspond to answers
y_test = np.argmax(y_test, axis=1)

# list of labels predicted from X_test
predicted_list = []

# for each array, extract largest element, find its index,
# and add the index to predicted_list. This index, like labels,
# would represent type of tumor
for element in prediction:
    element = element.tolist()
    answer = max(element)
    index = element.index(answer)
    predicted_list.append(index)


### prediction visualization (confusion matrix) ###
# print the confusion matrix where row is the 
# actual and column is the predicted to visualize
# prediction results
confusion = confusion_matrix(y_true=y_test, y_pred=predicted_list)

# visualize the confusion matrix
# the line below is taken from this source: 
# https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea#:~:text=The%20confusion%20matrix%20is%20a,Positive%20and%20False%20Negative%20categories
# after the phrase:'Luckily, we can make it beautiful with a heatmap from the Seaborn library.'
sns.heatmap(confusion, annot=True)

# add title and labels for clarity
plt.title("CNN Predictions")
plt.xlabel("Predicted")
plt.ylabel("Expected")

plt.show()
