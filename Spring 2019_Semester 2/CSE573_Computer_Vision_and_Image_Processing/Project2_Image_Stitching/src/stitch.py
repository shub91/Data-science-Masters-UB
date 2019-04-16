"""
Image Stitching

The goal of this task is to experiment with image stitching methods. Given a set of photos, your program
should be able to stitch them into a panoramic photo. There are no restrictions regarding the method you
use to stitch photos into a panoramic photo.

Please keep in mind that the best solution may require transformation of some of the images in 3D, not just
a simple overlap and blending.

For this project, you can assume you will have at most 3 images that you need to stitch together and that the
overlap of any two will be at least 20%. You will have to determine the spatial arrangement of the images automatically.
While some of the most modern techniques may use a spherical projection of better panoramas, you are free to assume that
basic 2D Planer transformations are sufficient.
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import imutils
from imutils import paths
import sys
import os
from scipy.ndimage.morphology import distance_transform_edt
from scipy.spatial.distance import cdist
from scipy.linalg import svd,lstsq
import argparse

np.seterr(divide='ignore', invalid='ignore') # To ignore floating-point erros which happen during computation

# def parse_args():
# 	parser = argparse.ArgumentParser(description="cse 473/573 project 2.")
# 	parser.add_argument("", "", type=str, required=False,
# 		help="path to input directory of images to stitch")
# 	# parser.add_argument("-o", "--output", type=str, required=True,
# 	# 	help="path to the output image")
# 	# parser.add_argument("-c", "--crop", type=int, default=0,
# 	# 	help="whether to crop out largest rectangular region")
# 	args = vars(parser.parse_args())
# 	return args

def computeH(i_coord, f_coord):
	# print("[INFO] Computing H..."
	i_x = i_coord.flatten()[0::2]
	i_y = i_coord.flatten()[1::2]
	f_x = f_coord.flatten()[0::2]
	f_y = f_coord.flatten()[1::2]
	assert (len(i_x) == len(i_y))
	assert (len(f_x) == len(f_y))
	assert (len(i_x) == len(f_y))
	X = []
	for i in range(0, len(i_x)):
		X.append([i_x[i], i_y[i], 1, 0, 0, 0, ((-i_x[i]) * f_x[i]), ((-i_y[i]) * f_x[i])])
		X.append([0, 0, 0, i_x[i], i_y[i], 1, ((-i_x[i]) * f_y[i]), ((-i_y[i]) * f_y[i])])
	X = np.array(X)
	b = f_coord.flatten().T
	q, w, e, r, t, y, u, i = np.linalg.lstsq(X, b, rcond = None)[0]
	H = np.array([[q, w, e], [r, t, y], [u, i, 1]])
	return H

def computeBH(im1, im2):
	orb = cv2.ORB_create()
	kp1, dsc1 = orb.detectAndCompute(im1, None)
	kp2, dsc2 = orb.detectAndCompute(im2, None)
	matchedimg1, _, index_img1 = filter_matches(kp1, kp2, dsc1, dsc2)
	matched_imgcoord_1_1 = [keypt[0].pt for keypt in matchedimg1]
	matched_imgcoord_1_2 = [keypt[1].pt for keypt in matchedimg1]
	hor_stack_coordinates = np.hstack((matched_imgcoord_1_1, matched_imgcoord_1_2))
	return ransacH(np.array(hor_stack_coordinates))

def ransacH(matches):
	print("[INFO] Using RANSAC to find the best Homography...")
	C = 0.95 # Confidence value
	BH = np.zeros((3, 3))
	ind = np.arange(0, (matches.shape[0]))
	itr = 0
	inlier_maximum = 0
	inlier_best = 0

	while (itr < 100000):
		np.random.shuffle(ind)
		matchesrnd = matches[ind[:4]]
		m_1, m_2 = matchesrnd[:, :2], matchesrnd[:, 2:]
		H = computeH(m_2, m_1)
		tr_pts = np.ones(((matches.shape[0]), 3))
		tr_pts[:, :2] = matches[:, 2:]
		tr_matches = np.dot(H, tr_pts.T)
		t = np.zeros(tr_matches.shape)
		for i in range(3):
			t[i, :] = tr_matches[i, :] / tr_matches[2, :]
		tr_matches = t.T
		SDS = np.sqrt(((matches[:, :2] - tr_matches[:, :2])**2).sum(1))
		inliers = SDS[SDS < C].shape[0]
		if inliers > inlier_maximum:
			inlier_best = ind[SDS < C]
			inlier_maximum = inliers
			BH = H
		itr = itr + 1
	return BH

def filter_matches(kp1,kp2,dsc1,dsc2,th=2000):
	print("[INFO] Matching the descriptors and filtering the best matches...")
	R = []
	R_dsc=[]
	R_ind = []
	for i in range(len(dsc1)):
		for j in range(len(dsc2)):
			if(sum(np.subtract(dsc1[i],dsc2[j]))<th):
					R.append((kp1[i],kp2[j]))
					R_ind.append((i,j))
					R_dsc.append(dsc1[i])
	return R,R_dsc,R_ind

def img2_seq(matched_imgcoord_1, matched_imgcoord_2, image_1, image_2):
	print("[INFO] Ordering the images in sequence...")
	if ((np.mean(matched_imgcoord_1, axis = 0)[0]) > (np.mean(matched_imgcoord_2, axis=0)[0])):
		im2_no = 'end'
	else:
		im2_no = 'start'
	return im2_no

def transform(p1,p2,p3,start_h,last_h,im1,im3):
	p_2 = cv2.perspectiveTransform(p2,start_h )
	p_3 = cv2.perspectiveTransform(p3, last_h )
	pts = np.concatenate((p1, p_2, p_3), axis=0)
	[minx, miny] = np.int32(pts.min(axis=0).ravel() - 0.5)
	[maxx, maxy] = np.int32(pts.max(axis=0).ravel() + 0.5)
	shift = np.array([[1, 0, (-minx)], [0, 1, (-miny)], [0,0,1]])
	transformleft = cv2.warpPerspective(im1, shift.dot(start_h), (maxx - minx, maxy - miny))
	transformright = cv2.warpPerspective(im3, shift.dot(last_h), (maxx - minx, maxy - miny))
	return transformleft, transformright, minx, miny

def stitch_img_seq(im1, im2, im3):
	print("[INFO] Stiching all the three images in sequence...")
	x1,y1,z1 = im1.shape
	x2,y2,z2 = im2.shape
	x3,y3,z3 = im3.shape
	start_h = computeBH(im2, im1)
	last_h = computeBH(im2, im3)
	p1 = np.float32([[0,0], [0,x1], [y1, x1], [y1,0]]).reshape(-1,1,2)
	p2 = np.float32([[0,0], [0,x2], [y2, x2], [y2,0]]).reshape(-1,1,2)
	p3 = np.float32([[0,0], [0,x3], [y3, x3], [y3,0]]).reshape(-1,1,2)
	transformleft, transformright, minx, miny = transform(p1,p2,p3,start_h,last_h,im1,im3)
	output = transformleft + transformright
	output[(-miny):x1+(-miny),(-minx):y1+(-minx)] = im2
	return output

def shift(s,row_min,im1,im2,H2_1,shape):
	SM = np.array([[s,0,0],[0,s,0],[0,0,1]])
	shift_M = np.array([[1,0,0],[0,1,-row_min],[0,0,1]])
	M = np.matmul(SM,shift_M)
	warp_image2 = cv2.warpPerspective(im2, np.matmul(M,H2_1), shape)
	warp_image1 = cv2.warpPerspective(im1, M, shape)
	M1 = distance_transform_edt(warp_image1)
	M2 = distance_transform_edt(warp_image2)
	return warp_image1, M1, warp_image2, M2

def img2_transform(im1, im2, H2_1):
	print("[INFO] Applying Transformation and stitching using Homography matrix without cliping...")
	imz1, imw1, imd1 = im1.shape
	imz2, imw2, imd2 = im2.shape
	crn = np.array([[0,imw2,0,imw2],[0,0,imz1,imz1],[1,1,1,1]])
	corner_warped = np.matmul(H2_1, crn)
	corner_warped = corner_warped/corner_warped[2,:]
	warp_corner = np.ceil(corner_warped)
	row1 = im1.shape[0]
	col1 = im2.shape[1]
	row_max = max(row1,max(warp_corner[1,:]))
	row_min = min(1,min(warp_corner[1,:]))
	col_max = max(col1,max(warp_corner[0,:]))
	col_min = min(1,min(warp_corner[0,:]))
	S = (col_max-col_min)/(row_max-row_min)
	Width = im1.shape[1]
	Height = im1.shape[0]
	shape = (Width, Height)
	s = Width / (col_max-col_min)
	warp_image1, M1, warp_image2, M2 = shift(s,row_min,im1,im2,H2_1,shape)
	f_image = np.divide(np.add(np.multiply(warp_image1,M1), np.multiply(warp_image2,M2)), np.add(M1, M2))
	f_image = np.nan_to_num(f_image,0)
	f_image = np.uint8(f_image)
	return f_image

def img_stitch(img_name, dir):
	im2 = None
	orb = cv2.ORB_create()
	if len(img_name) == 3:
		image_1 = cv2.imread(os.path.join(dir,img_name[0]))
		image_2 = cv2.imread(os.path.join(dir,img_name[1]))
		image_3 = cv2.imread(os.path.join(dir,img_name[2]))
		kp_1, dsc_1 = orb.detectAndCompute(image_1, None)
		kp_2, dsc_2 = orb.detectAndCompute(image_2, None)
		kp_3, dsc_3 = orb.detectAndCompute(image_3, None)
		matched_img1, _, index_img1 = filter_matches(kp_1, kp_2, dsc_1, dsc_2)
		matched_img2, _, index_img2 = filter_matches(kp_1, kp_3, dsc_1, dsc_3)
		matched_img3, _, index_img3 = filter_matches(kp_3, kp_2, dsc_3, dsc_2)
		matched_imgcoord_1_1 = [keypt[0].pt for keypt in matched_img1]
		matched_imgcoord_1_2 = [keypt[1].pt for keypt in matched_img1]
		matched_imgcoord_2_1 = [keypt[0].pt for keypt in matched_img2]
		matched_imgcoord_2_3 = [keypt[1].pt for keypt in matched_img2]
		matched_imgcoord_3_3 = [keypt[0].pt for keypt in matched_img2]
		matched_imgcoord_3_2 = [keypt[1].pt for keypt in matched_img2]
		minindex = np.argmin([len(matched_img1), len(matched_img2), len(matched_img3)])
		if minindex == 0:
			im2 = image_3
			if img2_seq(matched_imgcoord_2_1, matched_imgcoord_2_3, image_1, image_3) == 'end':
				im1 = image_1
				im3 = image_2
			else:
				im1 = image_2
				im3 = image_1
		elif minindex == 1:
			im2 = image_2
			if img2_seq(matched_imgcoord_1_1, matched_imgcoord_1_2, image_1, image_2) == 'end':
				im1 = image_1
				im3 = image_3
			else:
				im1 = image_3
				im3 = image_1
		elif minindex == 2:
			im2 = image_1
			if img2_seq(matched_imgcoord_1_1, matched_imgcoord_1_2, image_1, image_2) == 'end':
				im1 = image_2
				im3 = image_3
			else:
				im1 = image_3
				im3 = image_2
		panaroma = stitch_img_seq(im1, im2, im3)
	elif len(img_name) ==2:
		image_1 = cv2.imread(os.path.join(dir,img_name[0]))
		image_2 = cv2.imread(os.path.join(dir,img_name[1]))
		# image_1, image_2 = image[0], image[1]
		kp_1, dsc_1 = orb.detectAndCompute(image_1, None)
		kp_2, dsc_2 = orb.detectAndCompute(image_2, None)
		m_kp, _, index_img = filter_matches(kp_1, kp_2, dsc_1, dsc_2)
		matched_imgcoord_1 = [keypt[0].pt for keypt in m_kp]
		matched_imgcoord_2 = [keypt[1].pt for keypt in m_kp]
		if img2_seq(matched_imgcoord_1, matched_imgcoord_2, image_1, image_2) == 'start':
			hor_stack_coordinates = np.hstack((matched_imgcoord_2, matched_imgcoord_1))
			z1_2 = ransacH(np.array(hor_stack_coordinates))
			panaroma = img2_transform(image_2, image_1, z1_2)
		else:
			hor_stack_coordinates = np.hstack((matched_imgcoord_1, matched_imgcoord_2))
			H2_1 = ransacH(np.array(hor_stack_coordinates))
			panaroma = img2_transform(image_1, image_2, H2_1)
	else:
		print("[INFO] Image Stitching Failed. Please provide 2 or 3 images")
		exit(0)
	return(panaroma)

def main():
	# args = parse_args()
	# print("[INFO] loading images...")
	# imagePaths = sorted(list(paths.list_images(args["images"])))
	# images = []
	#
	# for imagePath in imagePaths:
	# 	   image = cv2.imread(imagePath)
	# 	   images.append(image)
	panaroma = None
	img_name = [photo for photo in os.listdir(sys.argv[1]) if not photo.startswith('.')]
	panaroma = img_stitch(img_name, sys.argv[1])
	if panaroma is not None:
		print("[INFO] Cropping...")
		stitched = cv2.copyMakeBorder(panaroma, 10, 10, 10, 10,
			cv2.BORDER_CONSTANT, (0, 0, 0))
		gray = cv2.cvtColor(panaroma, cv2.COLOR_BGR2GRAY)
		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		c = max(cnts, key=cv2.contourArea)
		mask = np.zeros(thresh.shape, dtype="uint8")
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
		minRect = mask.copy()
		sub = mask.copy()

		# keep looping until there are no non-zero pixels left in the
		# subtracted image
		while cv2.countNonZero(sub) > 0:
			# erode the minimum rectangular mask and then subtract
			# the thresholded image from the minimum rectangular mask
			# so we can count if there are any non-zero pixels left
			minRect = cv2.erode(minRect, None)
			sub = cv2.subtract(minRect, thresh)

		# allocate memory for the mask which will contain the
		# rectangular bounding box of the stitched image region
		mask = np.zeros(thresh.shape, dtype="uint8")
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
		cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		c = max(cnts, key=cv2.contourArea)
		(x, y, w, h) = cv2.boundingRect(c)

		# use the bounding box coordinates to extract the our final
		# stitched image
		panaroma = panaroma[y:y + h, x:x + w]


		print("[INFO] The image is stored as panaroma.jpg in the ubdata directory ...")
		print("[INFO] Operation Completed...")
		cv2.imwrite(os.path.join(sys.argv[1],'panaroma.jpg'), panaroma)
		# cv2.imwrite(args["output"], panaroma)
		# cv2.imshow("Stitched", panaroma)
		# cv2.waitKey(0)

if __name__ == "__main__":
	main()
