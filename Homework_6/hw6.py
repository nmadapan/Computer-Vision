import os, sys, time
import numpy as np
import cv2
from os.path import join, basename, dirname, splitext

sys.path.append('..\\utils')
from helpers import *

texture = True

# img_path = r'Images\img1.jpg'
# img_path = r'Images\img2.jpg'
img_path = r'Images\img3.jpg'

out_dir = dirname(img_path)
img_name = splitext(basename(img_path))[0]
if(texture): out_dir = join(out_dir, img_name) + '_texture'
write_flag = True
display = False

######################
## RGB Segmentation ##
######################
if(not texture):
    img = cv2.imread(img_path)
    ch_names = ['blue', 'green', 'red']
    init_thresh, mask, foreground, background = otsu_iter(img, display = display, out_dir = out_dir, img_name = img_name, ch_names = ch_names, write_flag = write_flag)

if(texture):
    img = cv2.imread(img_path, 0)
    ch_names = ['3x3', '5x5', '7x7']
    print 'Creating the texture: '
    img_texture = create_img_texture(img)
    cv2.imwrite(join(out_dir, img_name+'_texture.jpg'), img_texture)
    bins = np.linspace(0, np.max(img_texture.flatten()), num = 257)
    img_texture = (np.digitize(img_texture, bins) - 1).astype(np.uint8)
    cv2.imshow('img_texture', img_texture)
    cv2.waitKey(0)
    init_thresh, mask, foreground, background = otsu_iter(img_texture, display = display, out_dir = out_dir, img_name = img_name, ch_names = ch_names, write_flag = write_flag)
    # img_cont = create_contours(np.logical_not(mask))
    # cv2.imshow('Image contours', img_cont)
    # cv2.waitKey(0)
