import sys, os
import numpy as np

sys.path.insert(0, os.path.join('..', 'utils'))
from helpers import *

import time

## Initialization
all_img_paths = [os.path.join('.','images','1.jpg'), os.path.join('.','images','2.jpg'), os.path.join('.','images','3.jpg'), os.path.join('.','images','Seinfeld.jpg')]

img_path = os.path.join('.','images','n1.jpg')
img_path_ref = os.path.join('.','images','n2.jpg')

# img_path = os.path.join('.','images','edgar.png')
# img_path_ref = os.path.join('.','images','edgar_patch.jpg')

# img_path = all_img_paths[0]
# img_path_ref = all_img_paths[3]

pts = create_matching_points(img_path)['mps']
pts_ref = create_matching_points(img_path_ref)['mps']

## Find homography from 2 --> 1
H, _ = find_homography_2d(pts, pts_ref)
print H, H.shape

apply_trans_patch(img_path, img_path_ref, H)
apply_homography(img_path_ref, H, num_partitions = 1)
