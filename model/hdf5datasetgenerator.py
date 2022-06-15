import numpy as np
import h5py

class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize):
        self.batchSize = batchSize

        self.db = h5py.File(dbPath, "r")
        self.numImages = self.db["labels"].shape[0]

    def generator(self, passes=np.inf):
        epochs = 0

        while epochs < passes:
            for i in np.arange(0, self.numImages, self.batchSize):
                images1 = self.db["images1"][i: i+self.batchSize]
                images2 = self.db["images2"][i: i+self.batchSize]
                labels = self.db["labels"][i: i+self.batchSize]

                images1 = images1.astype("float")/255.0
                images2 = images2.astype("float")/255.0

                yield (np.array([images1, images2]), labels)
            
            epochs += 1
    
    def close(self):
        self.db.close()
                