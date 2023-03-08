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
parameters = aruco.DetectorParameters_create()


with open('calib.json') as f:
	calib = json.load(f)
dist_coeffs = np.array(calib['dist_coeffs'])
camera_matrix = np.array(calib['camera_matrix'])



cap = cv2.VideoCapture(int(config['camera']['device']))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(config['camera']['width']))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config['camera']['height']))
focus = config['camera']['focus']
if focus == 'auto':
    print('Setting focus to "auto"')
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
else:
    print(f'Setting focus to "manual" ({focus})')
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, int(focus))


while(True):
	ret, rawImage = cap.read()
	grayImage = cv2.cvtColor(rawImage, cv2.COLOR_BGR2GRAY)

	corners, ids, rejected = aruco.detectMarkers(grayImage, aruco_dict,
		parameters=parameters)

	if corners is not None and (len(corners) != 0):

		#aruco.drawDetectedMarkers(rawImage, corners, ids)

		diamond_corners, diamond_ids = aruco.detectCharucoDiamond(grayImage, \
			corners, ids, squareLengthMillimeters / markerLengthMillimeters)

		if diamond_corners is not None and (len(diamond_corners) != 0):

			aruco.drawDetectedDiamonds(rawImage, diamond_corners, diamond_ids)

			rvec, tvec, _objPoints = aruco.estimatePoseSingleMarkers(diamond_corners, \
				squareLengthMillimeters, camera_matrix, dist_coeffs)

			# In some OpenCV versions without _objPoints!
			#rvec, tvec = aruco.estimatePoseSingleMarkers(diamond_corners, \
			#	squareLengthMillimeters, camera_matrix, dist_coeffs)

			for i in range(len(tvec)):
				cv2.drawFrameAxes(rawImage, camera_matrix, dist_coeffs, \
					rvec[i], tvec[i], 20);

	cv2.imshow('Aruco diamond detect', rawImage)

	key = cv2.waitKey(1) & 0xff
	if key == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()

