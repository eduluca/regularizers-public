# USAGE
# python prepare_train_data.py
# import necessary packages
from utils import config
from imutils import paths
import numpy as np
from PIL import Image
import shutil
import os

data_train = np.load(config.TRAIN_DATA)
labels_train = np.load(config.TRAIN_LABELS)

corrected_labels_train = labels_train.copy()

changes = {
    0: [2400, 361, 2439, 1865, 2671, 893, 3625],
    1: [1288, 3033, 3037, 3599, 780, 3140, 3372],
    2: [1670, 673, 1995, 2804, 2769, 2714, 653],
    3: [2980, 2937, 603, 1819, 3704, 2025],
    4: [2665, 1649, 1777, 1563, 2986],
    5: [1587, 2750, 2763, 1917],
    6: [2015, 2388, 38, 401, 2818, 1875, 3675],
    7: [3154, 744, 452, 3283, 3412, 313],
    8: [556, 1213, 3253],
    9: [2940, 1314, 1108, 708, 3277, 728],
}
for new_label, indices_to_change in changes.items():
    for index in indices_to_change:
        corrected_labels_train[index] = new_label
        
labels_train = corrected_labels_train

labels_names = config.LABELS_NAMES

print("[INFO] saving data as images...")
# save the images first
for j in range (len(labels_names)):
    loc = np.where(labels_train==j)[0]
    if not os.path.exists('logo_photos/'+labels_names[j]):
        os.makedirs('logo_photos/'+labels_names[j]+'/')
    for i in range(len(loc)):
        idx = loc[i]
        img = data_train[:,idx].reshape((300,300,3))
        img = Image.fromarray(img, "RGB")
        image_filename = "c{}_{}.jpg".format(j,i)
        img.save("logo_photos/"+labels_names[j]+"/"+image_filename)

# Now copy them into train and val directory
def copy_images(imagePaths, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for path in imagePaths:
        # grab image name and its label from the path and create
        # a placeholder corresponding to the separate label folder
        imageName = path.split(os.path.sep)[-1]
        label = path.split(os.path.sep)[1]
        labelFolder = os.path.join(folder, label)
        if not os.path.exists(labelFolder):
            os.makedirs(labelFolder)
        # construct the destination path and copy the current image to it
        destination = os.path.join(labelFolder, imageName)
        shutil.copy(path, destination)
        
# load all the image paths and randomly shuffle them
print("[INFO] loading image paths...")
imagePaths = list(paths.list_images(config.DATA_PATH))
np.random.shuffle(imagePaths)
# generate training and validation paths
valPathsLen = int(len(imagePaths) * config.VAL_SPLIT)
trainPathsLen = len(imagePaths) - valPathsLen
trainPaths = imagePaths[:trainPathsLen]
valPaths = imagePaths[trainPathsLen:]
# copy the training and validation images to their respective directories
print("[INFO] copying training and validation images...")
copy_images(trainPaths, config.TRAIN)
copy_images(valPaths, config.VAL)