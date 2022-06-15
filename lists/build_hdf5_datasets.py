from model.hdf5datasetwriter import HDF5DatasetWriter
import cv2
import imutils
import pandas as pd
import numpy as np

def preprocess(image):
    # resize to 128x128x3
    # to grayscale
    
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
    writer = HDF5DatasetWriter((len(paths), 128, 128, 1), outputPath)

    for (path1, path2, label) in zip(paths1, paths2, labels):
        image = cv2.imread(path1)
        image = preprocess(image)

