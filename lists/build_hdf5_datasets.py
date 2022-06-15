from model.hdf5datasetwriter import HDF5DatasetWriter
from model.aspectawarepreprocessor import AspectAwarePreprocessor
import cv2
import imutils
import pandas as pd
import numpy as np

def preprocess(image):
    # resize to 128x128x3
    # to grayscale
    aap = AspectAwarePreprocessor(128, 128)
    image = aap.preprocess(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims(image, axis=-1)
    return image

TRAIN_HDF5 = "datasets/hdf5/train.hdf5"
VALID_HDF5 = "datasets/hdf5/val.hdf5"
TEST_HDF5 = "datasets/hdf5/test.hdf5"

train = pd.read_csv("lists/train.csv")
train_path1 = train["path1"]
train_path2 = train["path2"]
train_labels = train["label"]

test = pd.read_csv("lists/test.csv")
test_path1 = test["path1"]
test_path2 = test["path2"]
test_labels = test["label"]

validation = pd.read_csv("lists/val.csv")
validation_path1 = validation["path1"]
validation_path2 = validation["path2"]
validation_labels = validation["label"]

datasets = [
    (train_path1, train_path2, train_labels, TRAIN_HDF5),
    (validation_path1, validation_path2, validation_labels, VALID_HDF5),
    (test_path1, test_path2, test_labels, TEST_HDF5)]

for (paths1, paths2, labels, outputPath) in datasets:
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(paths1), 128, 128, 1), outputPath)

    for (path1, path2, label) in zip(paths1, paths2, labels):
        image1 = cv2.imread(path1)
        image1 = preprocess(image1)

        image2 = cv2.imread(path2)
        image2 = preprocess(image2)    

        writer.add([image1], [image2], [label])

