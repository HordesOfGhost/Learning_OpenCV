from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
from craft_text_detector import Craft
craft = Craft( crop_type="box", cuda=True,text_threshold=0.8,link_threshold=0.8,low_text=0.8)


cap = cv2.VideoCapture(0)

net = cv2.dnn.readNet("frozen_east_text_detection.pb")

def get_bounding_boxes(box):
	flat_box = box.flatten()
	x_min = round(min([flat_box[x] for x in [0,2,4,6]]))
	y_min = round(min([flat_box[y] for y in [1,3,5,7]]))
	x_max = round(max([flat_box[x] for x in [0,2,4,6]]))
	y_max = round(max([flat_box[y] for y in [1,3,5,7]]))
	return x_min,y_min,x_max,y_max

while cv2.waitKey(1) < 0:
	hasFrame, image = cap.read()
	orig = image
	(H, W) = image.shape[:2]

	(newW, newH) = (480, 240)
	rW = W / float(newW)
	rH = H / float(newH)

	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]


	boxes = craft.detect_text(image)['boxes']


	for box in boxes:

		x_min,y_min,x_max,y_max = get_bounding_boxes(box)

		# draw the bounding box on the image
		cv2.rectangle(orig, (x_min,y_min), (x_max,y_max), (255,0,0), 2)
	cv2.imshow("Text Detection", orig)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cv2.destroyAllWindows()
cap.release()