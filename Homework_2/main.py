import sys, os
import numpy as np

sys.path.insert(0, os.path.join('..', 'utils'))
from helpers import *

import time

# ## Initialization
all_img_paths = [os.path.join('.','images','1.jpg'), os.path.join('.','images','2.jpg'), os.path.join('.','images','3.jpg'), os.path.join('.','images','Jackie.jpg')]

# img_path = os.path.join('.','images','n4.jpg')
# img_path_ref = os.path.join('.','images','n6.jpg')

# img_path = os.path.join('.','images','edgar.png')
# img_path_ref = os.path.join('.','images','edgar_patch.jpg')

# img_path = all_img_paths[2]
# img_path_ref = all_img_paths[-1]

# pts = create_matching_points(img_path)['mps']
# pts_ref = create_matching_points(img_path_ref)['mps']

# ## Find homography from 2 --> 1
# H, _ = find_homography_2d(pts, pts_ref)

# print H

# # apply_trans_patch(img_path, img_path_ref, H)
# apply_trans_patch(img_path, img_path_ref, H)

# # apply_homography(img_path_ref, H, num_partitions = 4)
# # apply_homography(img_path, hinv(H), num_partitions = 4)


## Task 2
pts1 = create_matching_points(all_img_paths[0])['mps']
pts2 = create_matching_points(all_img_paths[1])['mps']
pts3 = create_matching_points(all_img_paths[2])['mps']

## Find homography from 2 --> 1
H12, _ = find_homography_2d(pts2, pts1) # 2 --> 1
H23, _ = find_homography_2d(pts3, pts2) # 3 --> 2
H13, _ = find_homography_2d(pts3, pts1) # 3 --> 1

_H13 = nmlz(np.dot(H12, H23))

print H13
print _H13

# apply_homography(all_img_paths[0], H12, num_partitions = 4, suff = '_H12')
apply_homography('.\\images\\1_H12.jpg', H23, num_partitions = 4, suff = '_H12_H23')

# apply_homography(all_img_paths[0], _H13, num_partitions = 4, suff = '_H12_H13')
