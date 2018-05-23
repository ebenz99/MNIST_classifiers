# Imports
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt


def main(args):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  np.save("trainImages", train_data, allow_pickle=True)
  np.save("trainLabels", train_labels, allow_pickle=True)
  np.save("evalImages", eval_data, allow_pickle=True)
  np.save("evalLabels", eval_labels, allow_pickle=True)


if __name__ == "__main__":
  tf.app.run()


