import sys, os
import numpy as np
from helpers import *

## Initialization
all_img_paths = ['.\\images\\1.jpg', '.\\images\\2.jpg', '.\\images\\3.jpg', '.\\images\\Seinfeld.jpg']

# img_path = '.\\images\\n1.jpg'
# img_path_ref = '.\\images\\n2.jpg'

img_path = all_img_paths[0]
img_path_ref = all_img_paths[-1]

pts = create_matching_points(img_path)['mps']
pts_ref = create_matching_points(img_path_ref)['mps']

## Find homography from 2 --> 1
H, _ = find_homography_2d(pts, pts_ref)

# apply_trans_patch(img_path, img_path_ref, H)
apply_homography(img_path_ref, H)
