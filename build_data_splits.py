from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import json
import cv2
import csv

df = pd.read_csv("data_pairs.csv")

image_pair1_paths = df["image1_path"]
image_pair2_paths = df["image2_path"]
velocities = df["linear_velocity"]

image_tuples = []
for i in range(len(image_pair1_paths)):
    image_tuples.append((image_pair1_paths[i], image_pair2_paths[i]))

print("[INFO] constructing splits...")
split = train_test_split(image_tuples, velocities, test_size = 0.15, random_state=42)
(trainPaths, testValPaths, trainLabels, testValLabels) = split

split = train_test_split(testValPaths, testValLabels, test_size = 0.50, random_state=42)
(testPaths, valPaths, testLabels, valLabels) = split

print("[INFO] size of training set: {}".format(len(trainPaths)))
print("[INFO] size of validation set: {}".format(len(valPaths)))
print("[INFO] size of test set: {}".format(len(testPaths)))

datasets = [
    ("train", trainPaths, trainLabels, "lists/train.csv"),
    ("val", valPaths, valLabels, "lists/val.csv"),
    ("test", testPaths, testLabels, "lists/test.csv")]

# initialize the list of Red, Green, and Blue channel averages
(R, G, B) = ([], [], [])

for (dType, paths, labels, outputPath) in datasets:
    # open the output file for writing
    print("[INFO] building {}...".format(outputPath))

    # create CSV file and write header to it
    header = ['index', 'label', 'path1', 'path2']

    with open(outputPath, 'w', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)

    for (i, (path, label)) in enumerate(zip(paths,labels)):
        text = [i, label, path[0], path[1]]
        with open(outputPath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(text)    

        if dType == "train":
            for image_path in path:
                image = cv2.imread(image_path)
                (b, g, r) = cv2.mean(image)[:3]
                R.append(r)
                G.append(g)            
                B.append(b)

print("[INFO] serializing means")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f.open("lists/dataset_mean.json")
f.write(json.dumps(D))
f.close()
    