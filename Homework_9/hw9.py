import cv2
import numpy as np
from helpers import *

left = {}
right = {}

left_path = 'pair1\\left.jpg'
right_path = 'pair1\\right.jpg'

left['mps'] = create_matching_points(left_path, suff = '')['mps']
right['mps'] = create_matching_points(right_path, suff = '')['mps']

left['img'] = cv2.imread(left_path)
right['img'] = cv2.imread(right_path)
l_height, l_width, _ = left['img'].shape
r_height, r_width, _ = right['img'].shape

F = compute_fund_mat(left['mps'], right['mps'], left['img'].shape, right['img'].shape)
if(F is None): sys.exit('Error! F is None')

F, e_left, e_prime, P, P_prime, G = compute_global_coordinates(left, right, F)

left['e'] = e_left
left['P'] = P

right['e'] = e_prime
right['P'] = P_prime

left_rect, right_rect, F_rect = rectify_images(left, right, F)

print 'Left'
print nmlz(left_rect['H'])

print 'Right'
print nmlz(right_rect['H'])

H_left = nmlz(left_rect['H'])
H_right = nmlz(right_rect['H'])

apply_homography(left['img'], H_left, suff = '_left')
apply_homography(right['img'], H_right, suff = '_right')

