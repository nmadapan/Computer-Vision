import numpy as np
import cv2
import os

def save_mps(event, x, y, flags, param):
    fac, mps = param
    if(event == cv2.EVENT_LBUTTONUP):
        mps.append([int(fac*x), int(fac*y)])
        print(int(fac*x), int(fac*y))

def create_matching_points(img_path):
    npz_path = img_path[:-4]+'.npz'
    flag = os.path.isfile(npz_path)
    if(not flag):
        img = cv2.imread(img_path)
        fac = max(float(int(img.shape[1]/960)), float(int(img.shape[0]/540)))
        resz_img = cv2.resize(img, None, fx=1/fac, fy=1/fac, interpolation = cv2.INTER_CUBIC)
        cv2.namedWindow('TempImage')
        mps = []
        cv2.setMouseCallback('TempImage', save_mps, param=(fac, mps))
        cv2.imshow('TempImage', resz_img)
        cv2.waitKey(0)
        np.savez(npz_path, mps = np.array(mps), info = {'x_max': img.shape[1], 'y_max': img.shape[0]})
    return np.load(npz_path)

## Initialization
img_path1 = '.\\images\\1.jpg'
pts1 = np.array([[2108, 423],[3310, 537],[3309, 1357],[2150, 1494]])

out = create_matching_points(img_path1)
mps, img_info = out['mps'], out['info']

# img = cv2.imread(img_path1)

# if(not os.path.isfile(img_path1[:-4]+'.npz')):
#     print(img.shape)
#     frame = cv2.resize(img, None, fx=1/fac, fy=1/fac, interpolation = cv2.INTER_CUBIC)
#     cv2.namedWindow('Image-1')
#     cv2.setMouseCallback('Image-1', func)
#     cv2.imshow('Image-1', frame)
#     cv2.waitKey(0)

# img_path2 = '.\\images\\2.jpg'
# pts2 = np.array([[1585,800],[2975,773],[3000,1520],[1620,1600]])

# img_path3 = '.\\images\\3.jpg'
# pts3 = np.array([[990,565], [2425,415], [2400,1480],[1020,1400]])

# img_path_ref = '.\\images\\Seinfeld.jpg'
# pts_ref = np.array([[0,0],[2560, 0],[2560, 1536],[0, 1536]])
