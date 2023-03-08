import cv2
import cv2.aruco as aruco
import glob
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


allCorners = []
allIds = []

imageSize = None

#for fname in glob.glob('Keyence_CA-LHR8/*.png'):
#for fname in glob.glob('Keyence_CA-LHR35/*.png'):
for fname in glob.glob('01/*.png'):
#for fname in glob.glob('Samyang_8mm/*.JPG'):
	print('Calibration using ' + fname + ' ...')

	img = cv2.imread(fname)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	imageSize = gray.shape

	corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict,
		parameters=parameters)

	aruco.drawDetectedMarkers(img, corners, ids)

	corners, ids, rejected, recovered_ids = aruco.refineDetectedMarkers( \
		gray, board, corners, ids, rejected)

	charuco_retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco( \
		corners, ids, gray, board)
	allCorners.append(charuco_corners)
	allIds.append(charuco_ids)

	aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)

	img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
	cv2.imshow('image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


print('Calculating calibration ...')
flags = 0
#flags = cv2.CALIB_FIX_ASPECT_RATIO
#flags = cv2.CALIB_RATIONAL_MODEL
reprojection_error, camera_matrix, dist_coeffs, rvecs, tvecs = \
	cv2.aruco.calibrateCameraCharuco(allCorners, allIds, \
	board, imageSize, None, None, flags=flags)
print('Reprojection error is {0:.2f}'.format(reprojection_error))
print('Camera matrix is ')
print(camera_matrix)
print('Distortion coefficients')
print(dist_coeffs)


calib = {}
calib['reprojection_error'] = reprojection_error
calib['camera_matrix'] = camera_matrix.tolist()
calib['dist_coeffs'] = dist_coeffs.tolist()
with open('calib.json', 'w') as f:
	json.dump(calib, f, indent=4, sort_keys=True)
