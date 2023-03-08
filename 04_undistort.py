import cv2
import json
import numpy as np


with open('calib.json') as f:
	calib = json.load(f)
dist_coeffs = np.array(calib['dist_coeffs'])
camera_matrix = np.array(calib['camera_matrix'])


image = cv2.imread('01/image0000.png')
#image = cv2.imread('Samyang_8mm/DSCF2282.JPG')
alpha = 1.0
h,  w = image.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix,
	dist_coeffs, (w,h), alpha, (w,h))

undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)


# Crop image
#x,y,w,h = roi
#undistorted_image = undistorted_image[y:y+h, x:x+w]


# Scaling of displayed images
#image = cv2.resize(image, (0,0), fx=0.2, fy=0.2)
#undistorted_image = cv2.resize(undistorted_image, (0,0), fx=0.2, fy=0.2)


while True:
	cv2.imshow('Image', image)
	if cv2.waitKey(2000) & 0xFF == ord('q'):
		break
	cv2.imshow('Image', undistorted_image)
	if cv2.waitKey(2000) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()
