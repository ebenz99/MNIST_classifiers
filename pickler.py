from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import pickle

def cnn_model_fn(features, labels, mode):
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])


def main(args):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images # Returns np.array

  np.save("trainImages", train_data, allow_pickle=True)
  np.save("trainLabels", train_labels, allow_pickle=True)
  np.save("evalImages", eval_data, allow_pickle=True)

  '''
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
  # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},y=train_labels,batch_size=20,num_epochs=None,shuffle=True)
  mnist_classifier.train(input_fn=train_input_fn,steps=1000,hooks=[logging_hook])
  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},y=eval_labels,num_epochs=1,shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)
  '''


if __name__ == "__main__":
  tf.app.run()


