# -*- coding: utf-8 -*-
"""
Created on Wed May 13 21:39:03 2020

@author: shubh
"""

import os
import random
from imutils import paths
import matplotlib.pyplot as plt
import shutil
from skimage import transform
from skimage import exposure
from skimage import io
import numpy as np
from resnet import ResNet
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD

from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
import argparse

"""
image_path_p=list(paths.list_images("cell_images/Parasitized/"))
random.seed(42)
random.shuffle(image_path_p)

i = int(len(image_path_p)*0.8)
train_path_p = image_path_p[:i]
test_path_p = image_path_p[i:]

i = int(len(train_path_p)*0.1)
len(train_path_p) = image_path_p[i:]
val_path_p = image_path_p[:i]

image_path_i=list(paths.list_images("cell_images/Uninfected/"))
random.seed(42)
random.shuffle(image_path_i)

i = int(len(image_path_i)*0.8)
train_path_i = image_path_i[:i]
test_path_i = image_path_i[i:]

i = int(len(train_path_i)*0.1)
train_path_i = image_path_i[i:]
val_path_i = image_path_i[:i]

os.makedirs("Train_p")
os.makedirs("Train_i")

os.makedirs("Test_p")
os.makedirs("Test_i")

os.makedirs("Validation_p")
os.makedirs("Validation_i")


for inputPath in val_path_i:
    labelPath = "Validation_i/"
    if os.path.exists(inputPath):
        shutil.copy2(inputPath,labelPath)
import pathlib

def load_split(basePath_p,basePath_i,label):
    data = list()
    labels = []
    rows_p="Validation_i"
    label="uninffected"
    for row in listdir(rows_p):
        #if os.path.exists(row):
            images = image.imread('Validation_i/' + row)

		# resize the image to be 32x32 pixels, ignoring aspect ratio,
		# and then perform Contrast Limited Adaptive Histogram
		# Equalization (CLAHE)
            images = transform.resize(images, (64, 64))

            data.append(images)
            labels.append(label)
    trainX=data
    trainY=labels
    testX=data
    testY=labels
    valX=data
    valY=labels
    

	# convert the data and labels to NumPy arrays
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)
    valX = np.array(valX)
    valY = np.array(valY)


	# return a tuple of the data and labels
    return (data, labels)
from os import listdir    
(trainX, trainY) = load_split("Train_p","Train_i","Parasitized")
(testX, testY) = load_split("Test_p/","Test_i/","Parasitized")
(valX, valY) = load_split("Validation_p/","Validation_i/","Parasitized")

plt.imshow(trainX[21000])
len(data)
"""
model = ResNet.build(64, 64, 3, 2, (1, 2, 4),(64, 128, 256, 512), reg=0.0005)

len(trainX) = trainX.astype("float32") / 255.0
valX = valX.astype("float32") / 255.0

NUM_EPOCHS = 20
INIT_LR = 1e-1
BS = 32

def poly_decay(epoch):
	# initialize the maximum number of epochs, base learning rate,
	# and power of the polynomial
	maxEpochs = NUM_EPOCHS
	baseLR = INIT_LR
	power = 1.0

	# compute the new learning rate based on polynomial decay
	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

	# return the new learning rate
	return alpha

totalTrain = len(list(paths.list_images("train/inffected/"))) + len(list(paths.list_images("train/uninffected/")))
totalVal = len(list(paths.list_images("val/inffected/"))) + len(list(paths.list_images("Val/uninffected/")))
totalTest = len(list(paths.list_images("test/inffected/"))) + len(list(paths.list_images("test/uninffected/")))



opt = SGD(lr=INIT_LR, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# define our set of callbacks and fit the model
callbacks = [LearningRateScheduler(poly_decay)]

# initialize the training training data augmentation object
Train_path = "train/"
Val_path = "val/"
Test_path = "examples"
# initialize the training training data augmentation object
trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=20,
	zoom_range=0.05,
	width_shift_range=0.05,
	height_shift_range=0.05,
	shear_range=0.05,
	horizontal_flip=True,
	fill_mode="nearest")

# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

# initialize the training generator
trainGen = trainAug.flow_from_directory(
	Train_path,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=True,
	batch_size=BS)

# initialize the validation generator
valGen = valAug.flow_from_directory(
	Val_path,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)


# initialize the testing generator
testGen = valAug.flow_from_directory(
	Test_path,
	class_mode=None,
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# initialize our ResNet model and compile it
model = ResNet.build(64, 64, 3, 2, (1, 2, 4),
	(64, 128, 256, 512), reg=0.0005)

model.summary()
opt = SGD(lr=INIT_LR, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# define our set of callbacks and fit the model
callbacks = [LearningRateScheduler(poly_decay)]
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // BS,
	validation_data=valGen,
	validation_steps=totalVal // BS,
	epochs=NUM_EPOCHS,
	callbacks=callbacks)




model.save("output/malaria_detection.model/")
# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
	steps=1)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys()))

N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("output/")

Labelnames= ["Inffected","Uninffected"]

labels=list()

for i in predIdxs:
    image = Labelnames[i]
    labels.append(image)
import cv2
imagePaths = list(paths.list_images("examples/test/"))
import imutils
for (i, imagePath) in enumerate(imagePaths):
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=128)
	cv2.putText(image, labels[i], (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
		0.45, (0, 0, 255), 2)

	# save the image to disk
	p = os.path.sep.join(["out/", "{}.png".format(i)])
	cv2.imwrite(p, image)