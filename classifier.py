'''
Classifier is trained using features extracted by CNN
Two classes: tampered and authentic images
SVM is used
'''

import numpy as np
from sklearn.utils import shuffle
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from sklearn.externals import joblib

if __name__ == '__main__':
	print('Reading data')
	tampered = np.load('tampered_train.npy')
	authentic = np.load('authentic_train.npy')

	zeros = np.zeros(tampered.shape[0])
	ones = np.ones(authentic.shape[0])

	features = np.concatenate((tampered,authentic), axis=0)
	labels = np.concatenate((zeros,ones), axis=0)

	features, labels = shuffle(features, labels)
	"""
	param_grid = { 
	           "kernel" : ['poly', 'rbf'],
	           "gamma" : [1e-1, 1e-2, 1e-3, 1e-4],
	           "C" : [1, 10, 100, 1000]}

	grid = model_selection.GridSearchCV(estimator=svm.SVC(), param_grid=param_grid, scoring='f1', n_jobs=-1, verbose=3)
	print('Fitting grid')
	grid.fit(features, labels)

	params = grid.best_params_
	print(params)
	clf = svm.SVC(kernel=params['kernel'], C=params['C'], gamma=params['gamma'])
	"""
	clf = svm.SVC(kernel='rbf', C=1, gamma=0.01)
	print('Fitting classifier')
	clf.fit(features, labels)
	clf.score(features, labels)

	tampered = np.load('tampered_test.npy')
	authentic = np.load('authentic_test.npy')

	zeros = np.zeros(tampered.shape[0])
	ones = np.ones(authentic.shape[0])

	features = np.concatenate((tampered,authentic), axis=0)
	labels = np.concatenate((zeros,ones), axis=0)

	clf.score(features, labels)

	print('Testing')
	predictions = clf.predict(features)
	print('f1 score: ', metrics.f1_score(labels, predictions))
	print('accuracy score: ', metrics.accuracy_score(labels, predictions))

	filename = 'classifier.joblib.pkl'
	joblib.dump(clf, filename)