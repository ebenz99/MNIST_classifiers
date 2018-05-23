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

		self.compIms = {}
		for idx in range(0,10):
			self.compIms[idx] = np.zeros((28,28))

		for idx in range(len(self.labs)):
			tempArr = self.compIms[self.labs[idx]]
			self.compIms[self.labs[idx]] = np.add(tempArr,self.ims[idx])
		for key in self.compIms:
			print(self.compIms[key].shape)
			plt.imshow(self.compIms[key])
			plt.show()

	def findClosestMatch(self, image):
		oldSum = 1000000
		label = 10
		bestIdx = 0
		for x in range(0,10):
			im = self.ims[x]
			lab = self.labs[x]
			newSum = np.ndarray.sum(np.multiply(im,image))
			if newSum < oldSum:
				oldSum = newSum
				label = lab
				bestIdx = x
		return label


'''TRAINING'''

#turns images from batch 1 into np array of train images

labs1 = train_labels
ims1 = train_images.reshape(-1, 28, 28).astype("float32")

labs2 = test_labels
ims2 = test_images.reshape(-1, 28, 28).astype("float32")


TrainSet = pipeDream(ims1,labs1)


#for procedure
numRight = 0
newLabels = []
numToRead = 300

#runs nearest neighbor classification, compares resulting label to real label
for num in range(0,len(labs2)):
	#print(ims2[num].shape)
	#plt.imshow(ims2[num])
	#plt.show()
	newLab = TrainSet.findClosestMatch(ims2[num])
	newLabels.append(newLab)
	if newLab == labs2[num]:
		numRight += 1
	if num > numToRead:
		break

#displays accuracy
accuracy = numRight / numToRead
print(accuracy)
