import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
import json


with open('config.json') as f:
	config = json.load(f)

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
numSquaresX = config['calibration_board']['squares_x']
numSquaresY = config['calibration_board']['squares_y']
squareLengthMillimeters = config['calibration_board']['square_length_millimeters']
markerLengthMillimeters = config['calibration_board']['marker_length_millimeters']
board = aruco.CharucoBoard_create(numSquaresX, numSquaresY,
	squareLengthMillimeters, markerLengthMillimeters, aruco_dict)


squareLengthPixels = 400
markerLengthPixels = int((squareLengthPixels * markerLengthMillimeters) / squareLengthMillimeters)
resolutionDPI = squareLengthPixels / (squareLengthMillimeters / 25.4)
widthPix = numSquaresX * squareLengthPixels
widthMillimeters = numSquaresX * squareLengthMillimeters
heightPix = numSquaresY * squareLengthPixels
heightMillimeters = numSquaresY * squareLengthMillimeters

print(f'board has {numSquaresX}x{numSquaresY} fields')
print(f'square length is {squareLengthMillimeters} mm or {squareLengthPixels} pix')
print(f'marker length is {markerLengthMillimeters} mm or {markerLengthPixels} pix')
print(f'image size is {widthMillimeters}x{heightMillimeters} mm or {widthPix}x{heightPix} pix')
print(f'resolution is {round(resolutionDPI)} dpi')


img = board.draw((widthPix, heightPix))
cv2.imwrite('charuco_pattern.png', img)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
plt.show()
