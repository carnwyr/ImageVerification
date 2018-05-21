# Helps to create ground truth images for CASIA2 dataset
# Also generates patches to train the model
# Generated patches are taken from single spliced area
# Works only when original and tampered pics are of the same size

import numpy as np
import cv2
import os
from glob import glob
import random

# Script must be located in CASIA directory to work
Tp_pic_list = glob('Tp' + os.sep + 'Tp*')
Au_pic_list = glob('Au' + os.sep + 'Au*')
ground_truth_dir = 'Gt'
train_Au_dir = 'TrAu'
train_Tp_dir = 'TrTp'

background_index = [16, 19, 24]
spliced_index = [34, 39]
au_index = [3, 15]

image_size = 64

# Creates path
def append_to_path(count, path, original, r):
	new_path = path
	for i in range(3-len(str(count))):
		new_path += '0'
	new_path += str(count)
	if original:
		new_path = train_Tp_dir + os.sep + new_path
	else:
		new_path = train_Tp_dir + os.sep + new_path + '_' + str(r)
	new_path += '.png'
	return new_path

# Creates ordinary and rotated patches
def create_patches(count, path, img):
	patch_path = append_to_path(count, path, True, 0)
	cv2.imwrite(patch_path, img)
	M = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),90,1)
	rot = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
	patch_path = append_to_path(count, path, False, 1)
	cv2.imwrite(patch_path, rot)
	rot = cv2.warpAffine(rot,M,(img.shape[1],img.shape[0]))
	patch_path = append_to_path(count, path, False, 2)
	cv2.imwrite(patch_path, rot)
	rot = cv2.warpAffine(rot,M,(img.shape[1],img.shape[0]))
	patch_path = append_to_path(count, path, False, 3)
	cv2.imwrite(patch_path, rot)
	rot = np.transpose(img, (1, 0, 2))
	patch_path = append_to_path(count, path, False, 4)
	cv2.imwrite(patch_path, rot)
	rot = cv2.warpAffine(rot,M,(img.shape[1],img.shape[0]))
	rot = cv2.warpAffine(rot,M,(img.shape[1],img.shape[0]))
	patch_path = append_to_path(count, path, False, 5)
	cv2.imwrite(patch_path, rot)
	patch_path = append_to_path(count, path, False, 6)
	cv2.imwrite(patch_path, cv2.flip(img, 0))
	patch_path = append_to_path(count, path, False, 7)
	cv2.imwrite(patch_path, cv2.flip(img, 1))

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

	# Create a rectangle with tampered area
	x, y, w, h = cv2.boundingRect(cnt)

	step_x = image_size
	step_y = image_size
	
	if w%step_x != 0:
		fitw = step_x - (w % step_x)
		if (x + w + fitw) < gt.shape[1]:
			w = w + fitw

	if x-image_size//4 > 0:
		x = x-image_size//4
		w = w+image_size//4
	if x+image_size//4+w < gt.shape[1]:
		w = w+image_size//4

	if h%step_y != 0:
		fith = step_y - (h % step_y)
		if (y + h + fith) < gt.shape[0]:
			h = h + fith

	if y-image_size//4 > 0:
		y = y-image_size//4
		h = h+image_size//4
	if y+image_size//4+h < gt.shape[0]:
		h = h+image_size//4
	
	count = 1

	path = 'Tp_' + path2[6] + '_' + path2[spliced_index[0]:spliced_index[1]] + '_'

	# Generate blocks
	for row in np.arange(start=y, stop=y + h - step_y + 1, step=step_y//2):
		for col in np.arange(start = x, stop = x + w - step_x + 1, step=step_x//2):
			shift_y = random.randint(max(0,row-image_size//2), min(row+image_size//2,gt.shape[0]-image_size))
			shift_x = random.randint(max(0,col-image_size//2), min(col+image_size//2,gt.shape[1]-image_size))
			patch = gt[shift_y:shift_y+step_y, shift_x:shift_x+step_x]
			if np.average(patch) > 10 and np.average(patch) < 235:
				create_patches(count, path, img2[shift_y:shift_y+step_y, shift_x:shift_x+step_x])
				count += 1
	
	# Generate ground truth image
	gt_path = ground_truth_dir + os.sep + 'Gt_D_' + path2[spliced_index[0]:spliced_index[1]] + '.png'
	cv2.imwrite(gt_path, gt)

# Generate authentic patches
def extract(path):
	img = cv2.imread(path)

	step_x = image_size
	step_y = image_size

	sh = img.shape

	for i in range(1,51):
		y = random.randint(0, sh[0]-step_y)
		x = random.randint(0, sh[1]-step_x)
		new_path = train_Au_dir + os.sep + path[au_index[0]:au_index[1]] + '_' + str(i) + '.png'
		cv2.imwrite(new_path, img[y:y+step_y, x:x+step_x])


for pic in Tp_pic_list:
	au_name = pic[background_index[0]:background_index[1]] + '_' + pic[background_index[1]:background_index[2]]
	au_pic = glob('Au' + os.sep + '???' + au_name + '*')
	if len(au_pic) == 0:
		print(pic, au_name)
		continue
	if 'txt' not in au_pic[0]:
		generate(au_pic[0], pic)
for pic in Au_pic_list:
	extract(pic)