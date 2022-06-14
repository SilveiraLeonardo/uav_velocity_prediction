# import the necessary packages
from tensorflow.keras.datasets import mnist
from imutils import build_montages # to visually validate our image pairs
import numpy as np
import cv2

def make_pairs(images, labels):
	# initialize lists to hold the images tuple (image,image) and their labels
	pairImages = []
	pairLabels = []

	# calculate the total number of classes
	numClasses = len(np.unique(labels))
	# create a list of indexes for each class
	# a list of 10 lists with the indexes of the images for each class
	idx = [np.where(labels == i)[0] for i in range(0, numClasses)]

	for idxA in range(len(images)):
		# grab the current image
		currentImage = images[idxA]
		# and label
		label = labels[idxA]

		# randomly pick an image that belongs to the same class label
		idxB = np.random.choice(idx[label])
		posImage = images[idxB]

		# prepare a positive pair and update the images and labels lists
		pairImages.append([currentImage, posImage])
		pairLabels.append([1])

		negIdx = np.where(labels != label)[0]
		negImage = images[np.random.choice(negIdx)]

		# prepare a negative pair of images and update our lists
		pairImages.append([currentImage, negImage])
		pairLabels.append([0])

	# return a 2-tuple of image pairs and labels
	return (np.array(pairImages), np.array(pairLabels))

# load mnist dataset
print("[INFO] loading MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data()

# build the positive and negative image pairs
print("[INFO] preparing positive and negative pairs...")
(pairTrain, labelTrain) = make_pairs(trainX, trainY)
(pairTest, labelTest) = make_pairs(testX, testY)

# initialize the list of images that will be used when 
# building our montage
images = []

# loop over training pairs
for i in np.random.choice(np.arange(0, len(pairTrain)), size=(49,)):
	# grab image pair and label
	imageA = pairTrain[i][0]
	imageB = pairTrain[i][1]
	label = labelTrain[i]

	# to make it easier to visualize, we are going to pad the images
	output = np.zeros((36,60), dtype="uint8")
	pair = np.hstack([imageA, imageB])
	output[4:32, 0:56] = pair

	# annotate frame
	text = "neg" if label[0]==0 else "pos"
	color = (0,0,255) if label[0]==0 else (0,255,0)

	# create a 3-channel RGB image from the grayscale pair
	vis = cv2.merge([output]*3)
	# resize it for better visualization
	vis = cv2.resize(vis, (96,51), interpolation=cv2.INTER_LINEAR)
	# annotate it
	cv2.putText(vis, text, (2,12), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

	# add the pair to our list
	images.append(vis)	

# construct montage
montage = build_montages(images, (96,51), (7,7))[0]

# show output frame
cv2.imwrite("montage.png", montage)
