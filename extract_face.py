#!/usr/bin/env python3

import numpy
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description='Extract face from image')

parser.add_argument('--image', type=str,
					default='', help='Indicate the image filename')
parser.add_argument('--output', type=str,
					default='', help='Indicate the output for image cropped')
parser.add_argument('--haar', type=str,
					default='haarcascade_frontalface_alt.xml',
					help='Indicate the filename for haarcascade xml')

args = parser.parse_args()

face_cascade = cv2.CascadeClassifier(args.haar)

def cropFace(filename, output_filename):
	img = cv2.imread(filename)
	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(imgGray, 1.9, 2)
	if len(faces) > 0:
		maxes = faces[:2,-1]
		max_idx = numpy.argmax(maxes)

		faces = [faces[max_idx]]
		for (x,y,w,h) in faces:
			padding = int(max(x, y)*0.4)
			if padding > x+w:
				padding=int(max(x, y)*0.2)
			cv2.rectangle(img,(x-padding,y-padding),(x+w+padding,y+h+padding),(0,255,255),2)

			crop_img = img[y-padding:y+h+padding, x-padding:x+w+padding]
			prove_it = face_cascade.detectMultiScale(crop_img, 1.2, 5)
			if len(prove_it) > 0:
				cv2.imwrite(output_filename, crop_img)
	else:
		print("None face detected!")

if args.image != '' and args.output != '':
	cropFace(args.image, args.output)
else:
	parser.print_help()