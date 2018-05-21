'''
Module that extracts features from image
Picture is split into patches
Features are extracted from them and fused into final feature
Final features are saved into a file
'''

import os
import numpy as np
import tensorflow as tf
import cv2
from model import cnn_model_fn
from glob import glob

img_size = 64

descriptor = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="tampered_authentic_model")

def save_desc(img, isAuthentic):
	image = cv2.imread(img)
	image = image.astype(np.float32)
	image = np.multiply(image, 1.0 / 255.0)
	h, w, _ = image.shape
	patches = []
	for row in np.arange(start=0, stop= h - img_size + 1, step=img_size):
		for col in np.arange(start = 0, stop = w - img_size + 1, step=img_size):
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
	if isAuthentic == 0:
		final_tampered_features.append(fused_feature)
	else:
		final_authentic_features.append(fused_feature)

final_tampered_features = []
final_authentic_features = []

tampered = glob('training_pics' + os.sep + 'tampered' + os.sep + 'Tp*')
authentic = glob('training_pics' + os.sep + 'authentic' + os.sep + 'Au*')

for img in tampered:
	save_desc(img, 0)

np.save('tampered_train.npy', final_tampered_features)
tampered = []
final_tampered_features = []

for img in authentic:
	save_desc(img, 1)

np.save('authentic_train.npy', final_authentic_features)
authentic = []
final_authentic_features = []

tampered = glob('testing_pics' + os.sep + 'tampered' + os.sep + 'Tp*')
authentic = glob('testing_pics' + os.sep + 'authentic' + os.sep + 'Au*')

for img in tampered:
	save_desc(img, 0)

np.save('tampered_test.npy', final_tampered_features)
tampered = []
final_tampered_features = []

for img in authentic:
	save_desc(img, 1)

np.save('authentic_test.npy', final_authentic_features)
authentic = []
final_authentic_features = []