'''
Verifies a single image using trained classifier
'''

import sys
import os
import numpy as np
import tensorflow as tf
import cv2
from model import cnn_model_fn
from glob import glob
import csv
import telegram_send
from sklearn.externals import joblib

img_size = 64

descriptor = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="tampered_authentic_model")

image = cv2.imread(sys.argv[1])
image = image.astype(np.float32)
image = np.multiply(image, 1.0 / 255.0)
h, w, _ = image.shape
patches = []
for row in np.arange(start=0, stop= h - img_size + 1, step=img_size//2):
	for col in np.arange(start = 0, stop = w - img_size + 1, step=img_size//2):
		patches.append(image[row:row+img_size, col:col+img_size])
patches = np.array(patches)
input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": patches},
			batch_size=100,
			shuffle=False)
predictions = list(descriptor.predict(input_fn=input_fn))
features = [prediction['descriptor'] for prediction in predictions]
fused_feature = np.zeros(features[0].shape[0])
for i in range(features[0].shape[0]):
	elems = [feature[i] for feature in features]
	fused_feature[i] = max(elems)

clf = joblib.load('classifier.joblib.pkl')

predictions = clf.predict([fused_feature])
print(predictions)
if predictions[0] == 0:
	print("IT'S FAAAAKE!")
else:
	print("seems legit")