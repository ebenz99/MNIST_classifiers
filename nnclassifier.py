import pickle
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import random

import numpy as np
import os

train_images = np.load(os.getcwd()+"\\trainImages.npy")
train_labels = np.load(os.getcwd()+"\\trainLabels.npy")

test_images = np.load(os.getcwd()+"\\evalImages.npy")
test_labels = np.load(os.getcwd()+"\\evalLabels.npy")



#class to hold train data set and run comparisons from them
class pipeDream:
	def __init__(self, arrayOfImages, arrayOfLabels):
		self.ims = arrayOfImages
		self.labs = arrayOfLabels
	def findClosestMatch(self, image):
		dif = 1000000
		label = 10
		bestIdx = 0
		for x in range(len(self.labs)):
			im = self.ims[x]
			lab = self.labs[x]
			newDif = np.ndarray.sum(np.subtract(im,image))
			if newDif < dif:
				dif = newDif
				label = lab
				bestIdx = x
		return label


'''TRAINING'''

#turns images from batch 1 into np array of train images

labs1 = train_labels
ims1 = train_images.reshape(-1, 28, 28, 1).astype("uint8")

labs2 = test_labels
ims2 = test_images.reshape(-1, 28, 28, 1).astype("uint8")


TrainSet = pipeDream(ims1,labs1)

#for procedure
numRight = 0
newLabels = []
numToRead = 300

#runs nearest neighbor classification, compares resulting label to real label
for num in range(0,len(labs2)):
	newLab = TrainSet.findClosestMatch(ims2[num])
	newLabels.append(newLab)
	if newLab == labs2[num]:
		numRight += 1
	if num > numToRead:
		break

#displays accuracy
accuracy = numRight / numToRead
print(accuracy)
