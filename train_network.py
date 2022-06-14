# import the necessary packages
from model.model import build_siamese_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Flatten
from tensorflow.keras.losses import MeanSquaredError

def monitor_training(H):
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, len(H.history["loss"])), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, len(H.history["loss"])), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, len(H.history["loss"])), H.history["accuracy"], label="train_acc")
	plt.plot(np.arange(0, len(H.history["loss"])), H.history["val_accuracy"], label="val_accuracy")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend()
	plt.savefig("plots/monitor_{}.png".format(len(H.history["loss"])))

IMG_SHAPE = (240, 240, 3)
chanDim = -1
BATCH_SIZE = 64
EPOCHS = 100
MODEL_PATH = "model/velocity_pred_model"

# configure the siamese network
print("[INFO] build network...")
imgA = Input(shape=IMG_SHAPE)
imgB = Input(shape=IMG_SHAPE)
featureExtractor = build_siamese_model(IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

x = concatenate([featsA, featsB], axis=chanDim)
x = Conv2D(512, (3,3), padding="same", activation="relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = Conv2D(512, (3,3), padding="same", activation="relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = MaxPooling2D(pool_size=2)(x)
x = Dropout(0.2)(x)

x = Conv2D(512, (3,3), padding="same", activation="relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = Conv2D(512, (3,3), padding="same", activation="relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = MaxPooling2D(pool_size=2)(x)
x = Dropout(0.2)(x)

x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation="relu")(x)
x = BatchNormalization()(x)
output = Dense(1, activation="linear")(x)

model = Model(inputs=[imgA, imgB], outputs=output)

# best model checkpoint
ckp_path = "checkpoints/model_{epoch:02d}.hdf5"
mcp = tf.keras.callbacks.ModelCheckpoint(filepath=ckp_path,
										monitor="val_loss",
										save_best_only=True,
										mode="auto",
										save_freq="epoch",
										verbose=1)

print("[INFO] compiling model...")
model.compile(optimizer="adam", loss=MeanSquaredError(), metrics=["accuracy"])

print("[INFO] training model...")
history = model.fit(
	[pairTrain[:,0], pairTrain[:,1]], labelTrain[:],
	validation_data=([pairTest[:,0], pairTest[:,1]], labelTest[:]),
	batch_size = BATCH_SIZE,
	epochs= EPOCHS)

print("[INFO] serializing model")
model.save(MODEL_PATH)

print("[INFO] plotting training history")
monitor_training(history)