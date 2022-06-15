# import the necessary packages
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from model.siamese_network import build_siamese_model
from model.trainingMonitor import TrainingMonitor
from model.hdf5datasetgenerator import HDF5DatasetGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, help="path to model to load")
ap.add_argument("-s", "--start_epoch", type=int, default=0, help="epoch to restart training")
args = vars(ap.parse_args())

IMG_SHAPE = (128, 128, 1)
chanDim = -1
BATCH_SIZE = 64
EPOCHS = 100
MODEL_PATH = "checkpoints/velocity_pred_model"
FIG_PATH = "plots/monitor_{}.png".format(os.getpid())
TRAIN_HDF5 = "datasets/hdf5/train.hdf5"
VALID_HDF5 = "datasets/hdf5/val.hdf5"
TEST_HDF5 = "datasets/hdf5/test.hdf5"

if args["model"] is None:
    # configure the siamese network
    print("[INFO] build network...")
    imgA = Input(shape=IMG_SHAPE)
    imgB = Input(shape=IMG_SHAPE)
    featureExtractor = build_siamese_model(IMG_SHAPE)
    featsA = featureExtractor(imgA)
    featsB = featureExtractor(imgB)

    x = concatenate([featsA, featsB], axis=chanDim)
    x = Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = Conv2D(256, (3,3), padding="same", activation="relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Conv2D(256, (3,3), padding="same", activation="relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    output = Dense(1, activation="linear")(x)

    model = Model(inputs=[imgA, imgB], outputs=output)

    print("[INFO] compiling model...")
    opt = SGD(lr=1e-2, momentum=0.9)
    model.compile(loss= "mean_squared_error" , optimizer=opt, metrics=["mean_squared_error"])
else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])

    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-3)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr))))

trainGen = HDF5DatasetGenerator(TRAIN_HDF5, BATCH_SIZE)
valGen = HDF5DatasetGenerator(VALID_HDF5, BATCH_SIZE)

# best model checkpoint
ckp_path = "checkpoints/model_{epoch:02d}.hdf5"
mcp = ModelCheckpoint(filepath=ckp_path,
					monitor="val_loss",
					save_best_only=True,
					mode="auto",
					save_freq="epoch",
					verbose=1)

callbacks=[mcp, TrainingMonitor(FIG_PATH, startAt=args["start_epoch"])]

print("[INFO] training model...")
model.fit(
	trainGen.generator(),
	steps_per_epoch = trainGen.numImages // BATCH_SIZE,
	validation_data=valGen.generator(),
	validation_steps = valGen.numImages // BATCH_SIZE,
	epochs = EPOCHS,
	max_queue_size = 10,
	callbacks = callbacks,
	verbose = 1)

print("[INFO] serializing model")
model.save(MODEL_PATH, overwrite=True)

trainGen.close()
valGen.close()