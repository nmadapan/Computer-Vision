import cv2
import numpy as np
from helpers import *

left_path = 'pair1\\left.jpg'
right_path = 'pair1\\right.jpg'

left_mps = create_matching_points(left_path, suff = '')['mps']
right_mps = create_matching_points(right_path, suff = '')['mps']

left = cv2.imread(left_path)
right = cv2.imread(right_path)
l_height, l_width, _ = left.shape
r_height, r_width, _ = left.shape

F = compute_fund_mat(left_mps, right_mps, left.shape, right.shape)
if(F is None): sys.exit('Error! F is None')

F, P_prime, G = compute_global_coordinates(F, left_mps, right_mps, left.shape, right.shape)
