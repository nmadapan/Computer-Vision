import os, sys, time
import numpy as np
import cv2

sys.path.append('..\\utils')
from helpers import *

# img_path = r'Images\lighthouse.jpg')
img_path = r'Images\baby.jpg'
img_path = r'Images\ski.jpg'

img = cv2.imread(img_path, 0)

img_texture = create_img_texture(img)
bins = np.linspace(0, np.max(img_texture.flatten()), num = 257)
img_texture = (np.digitize(img_texture, bins) - 1).astype(np.uint8)
cv2.imshow('img_texture', img_texture)
cv2.waitKey(0)
init_thresh, mask, foreground, background = otsu_iter(img_texture, num_iter = 2, display = False)
img_cont = create_contours(np.logical_not(mask))
cv2.imshow('Image contours', img_cont)
cv2.waitKey(0)

# otsu_channels(img, display = False)
# init_thresh, mask, foreground, background = otsu_iter(img, num_iter = 1, display = False)
# img_cont = create_contours(mask)
# cv2.imshow('Image contours', img_cont)
# cv2.waitKey(0)

