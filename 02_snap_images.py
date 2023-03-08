import numpy as np
import copy
import cv2
import os
import json


with open('config.json') as f:
	config = json.load(f)

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

savePath = '01'
imageNo = 0


while(True):
	ret, rawImage = cap.read()

	displayImage = copy.copy(rawImage)

	cv2.putText(displayImage,'Next: image{0:04d}.png'.format(imageNo),
		(30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

	cv2.imshow('Press "s" to save and "q" to quit', displayImage)

	key = cv2.waitKey(1) & 0xff
	if key == ord('q'):
		break
	elif key == ord('s'):
		filename = os.path.join(savePath,
			'image{0:04d}.png'.format(imageNo))
		cv2.imwrite(filename, rawImage)
		imageNo += 1


cap.release()
cv2.destroyAllWindows()
