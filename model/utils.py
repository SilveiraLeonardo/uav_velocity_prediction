from tensorflow import keras
import numpy as np
import imutils
import cv2

# inspired by: https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71#:~:text=Note%3A%20As%20our%20dataset%20is,disk%20in%20batches%20to%20memory.

class Custom_Generator(keras.utils.Sequence):

    def __init__(self, image_paths, labels, batch_size, rMean, gMean, bMean, image_size=480):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean
        self.size = image_size

    def __len__(self):
        return (np.ceil(len(self.image_paths)/float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_paths[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.labels[idx*self.batch_size:(idx+1)*self.batch_size]

        processed_batch_x = []
        for pair in batch_x:
            image1 = self.MeanProcessor(imutils.resize(cv2.imread(pair[0])), width=self.size)
            image2 = self.MeanProcessor(imutils.resize(cv2.imread(pair[1])), width=self.size)

            processed_batch_x.append([image1, image2])

        return np.array(processed_batch_x), np.array(batch_y)
