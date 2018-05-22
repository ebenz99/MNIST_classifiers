import numpy as np
import os

train_images = np.load(os.getcwd()+"\\trainImages.npy")
train_labels = np.load(os.getcwd()+"\\trainLabels.npy")

test_images = np.load(os.getcwd()+"\\evalImages.npy")
test_labels = np.load(os.getcwd()+"\\evalLabels.npy")



