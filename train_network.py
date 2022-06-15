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
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# def monitor_training(H):
# 	plt.style.use("ggplot")
# 	plt.figure()
# 	plt.plot(np.arange(0, len(H.history["loss"])), H.history["loss"], label="train_loss")
# 	plt.plot(np.arange(0, len(H.history["loss"])), H.history["val_loss"], label="val_loss")
# 	plt.plot(np.arange(0, len(H.history["loss"])), H.history["accuracy"], label="train_acc")
# 	plt.plot(np.arange(0, len(H.history["loss"])), H.history["val_accuracy"], label="val_accuracy")
# 	plt.title("Training Loss and Accuracy")
# 	plt.xlabel("Epoch #")
# 	plt.ylabel("Loss/Accuracy")
# 	plt.legend()
# 	plt.savefig("plots/monitor_{}.png".format(len(H.history["loss"])))

IMG_SHAPE = (128, 128, 1)
chanDim = -1
BATCH_SIZE = 64
EPOCHS = 100
MODEL_PATH = "checkpoints/velocity_pred_model"
FIG_PATH = "plots/monitor_{}.png".format(os.getpid())
TRAIN_HDF5 = "datasets/hdf5/train.hdf5"
VALID_HDF5 = "datasets/hdf5/val.hdf5"
TEST_HDF5 = "datasets/hdf5/test.hdf5"

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

callbacks=[mcp, TrainingMonitor(FIG_PATH)]

print("[INFO] compiling model...")
model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])

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