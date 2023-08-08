# USAGE
# python test.py --model output/finetune_model.pth
# import the necessary packages
from utils import config
from utils import create_dataloaders
from torchvision import transforms
from torch import nn
import argparse
import torch
import os
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report

print("[INFO] loading test data...")
data_test = np.load(config.TEST_DATA)
labels_test = np.load(config.TEST_LABELS)
labels_names = config.LABELS_NAMES

# save test images
print("[INFO] saving data as images in dataset/test/ directory...")
for j in range (len(labels_names)):
    loc = np.where(labels_test==j)[0]
    if not os.path.exists("dataset/test/"+labels_names[j]):
        os.makedirs("dataset/test/"+labels_names[j]+"/")
    for i in range(len(loc)):
        idx = loc[i]
        img = data_test[:,idx].reshape((300,300,3))
        img = Image.fromarray(img, "RGB")
        image_filename = "c{}_{}.jpg".format(j,i)
        img.save("dataset/test/"+labels_names[j]+"/"+image_filename)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="output/finetune_model.pth", help="path to trained model model")
args = vars(ap.parse_args())

# build our data pre-processing pipeline
testTransform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD)
])

# initialize our test dataset and data loader
print("[INFO] loading data into dataloader...")
(testDS, testLoader) = create_dataloaders.get_dataloader(config.VAL,
    transforms=testTransform, batchSize=len(labels_test), shuffle=True)

# check if we have a GPU available, if so, define the map location accordingly
if torch.cuda.is_available():
    map_location = lambda storage, loc: storage.cuda()
# otherwise, we will be using CPU to run our model
else:
    map_location = "cpu"
# load the model
print("[INFO] loading the model...")
model = torch.load(args["model"], map_location=map_location)
# move the model to the device and set it in evaluation mode
model.to(config.DEVICE)
model.eval()

# grab a batch of test data
batch = next(iter(testLoader))
(images, labels) = (batch[0], batch[1])

pred_list = []
gt_list = []
# switch off autograd
with torch.no_grad():
    # send the images to the device
    images = images.to(config.DEVICE)
    # make the predictions
    print("[INFO] performing inference...")
    preds = model(images)
    
    for i in range(0, len(labels_test)):
        gt = labels[i].cpu().numpy()
        gt_list.append(gt)
        pred = preds[i].argmax().cpu().numpy()
        pred_list.append(pred)

    print("[INFO] done")
    print(classification_report(gt_list, pred_list))