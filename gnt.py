# Helps to create ground truth images for CASIA2 dataset
# Also generates patches to validate the model
# Generated patches are taken from single spliced area
# Works only when original and tampered pics are of the same size

import numpy as np
import cv2
import os
from glob import glob
import random

# Script must be located in CASIA directory to work
Tp_pic_list = glob('testing_pics' + os.sep + 'tampered' + os.sep + 'Tp*')
Au_pic_list = glob('testing_pics' + os.sep + 'authentic' + os.sep + 'Au*')
Gt_test_dir = 'Gt_test'
testing_dir = 'testing_data_new'

background_index = [35, 38, 43]
spliced_index = [53, 58]
au_index = [23, 35]

image_size = 64

# Creates ordinary and rotated patches
def create_patch(count, path, directory, img):
	patch_path = path
	for i in range(3-len(str(count))):
		patch_path += '0'
	patch_path += str(count)
	patch_path = testing_dir + os.sep + directory + os.sep + patch_path
	patch_path += '.png'
	cv2.imwrite(patch_path, img)

# Recieve original and spliced image paths
# Generate ground truth image
# Generate blocks of spliced image with maniputated parts
def generate(path1, path2):
	img1 = cv2.imread(path1)
	img2 = cv2.imread(path2)
	
	if img1.shape != img2.shape:
		return

	# Find difference between images and convert to bw
	diff = cv2.absdiff(img1, img2)
	mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

	# Binarize diff image (create mask)
	(_,th) = cv2.threshold(mask,5,255,cv2.THRESH_BINARY)

	# Dismiss bright pictures because of high chance of mistake
	if np.average(th) > 230:
		return

	# Find contours
	# Draw big objects
	# Find the largest contour
	(_, cnts, _) = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	gt = np.zeros(th.shape, np.uint8)
	for c in cnts:
		if cv2.contourArea(c) > 50:
			cv2.drawContours(gt, [c], 0, (255, 255, 255), cv2.FILLED)

	areas = [cv2.contourArea(c) for c in cnts]
	max_index = np.argmax(areas)
	cnt=cnts[max_index]

	# Return if nothing was found
	if cv2.contourArea(cnt) <= 50:
		return

	x = 0
	y = 0
	w = gt.shape[1]
	h = gt.shape[0]

	step_x = image_size
	step_y = image_size
	
	count_t = 1
	count_a = 1

	path = 'Tp_' + path2[25] + '_' + path2[spliced_index[0]:spliced_index[1]] + '_'

	# Generate blocks
	for row in np.arange(start=y, stop=y + h - step_y + 1, step=step_y):
		for col in np.arange(start = x, stop = x + w - step_x + 1, step=step_x):
			patch = gt[row:row+step_y, col:col+step_x]
			if np.average(patch) > 5 and np.average(patch) < 250:
				create_patch(count_t, path, 'tampered', img2[row:row+step_y, col:col+step_x])
				count_t += 1

# Generate authentic patches
def extract(path):
	img = cv2.imread(path)

	step_x = image_size
	step_y = image_size

	sh = img.shape

	for i in range(1,5):
		y = random.randint(0, sh[0]-step_y)
		x = random.randint(0, sh[1]-step_x)
		new_path = 'testing_data_new' + os.sep + 'authentic' + os.sep + path[au_index[0]:au_index[1]] + '_' + str(i) + '.png'
		cv2.imwrite(new_path, img[y:y+step_y, x:x+step_x])


for pic in Tp_pic_list:
	au_name = pic[background_index[0]:background_index[1]] + '_' + pic[background_index[1]:background_index[2]]
	au_pic = glob('Au' + os.sep + '???' + au_name + '*')
	if len(au_pic) == 0:
		au_pic = glob('testing_pics'+ os.sep + 'authentic' + os.sep + '???' + au_name + '*')
		if len(au_pic) == 0:
			print(pic, au_name)
			continue
	if 'txt' not in au_pic[0]:
		generate(au_pic[0], pic)
for pic in Au_pic_list:
	extract(pic)