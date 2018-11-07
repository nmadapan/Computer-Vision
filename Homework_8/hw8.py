import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist

img = cv2.imread(r'Data\Dataset1\Pic_1.jpg',0)

edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

# plt.show()

lines = cv2.HoughLines(edges,0.8,np.pi/180,40)
print lines.shape

lines = np.reshape(lines, (lines.shape[0], lines.shape[2]))

M = pdist(lines, metric = 'euclidean')
print np.min(M), np.max(M)

for line in lines:
    rho, theta = line[0], line[1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow('Lines', img)

cv2.waitKey(0)
