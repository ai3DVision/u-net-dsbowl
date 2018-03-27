import numpy as np
import cv2
import os

if __name__ == "__main__":
	DATA_PATH = "data/train/"
	for fn in os.listdir(DATA_PATH):
		masks_path = DATA_PATH + fn + "/masks/"
		img = cv2.imread(masks_path + "../images/" + fn + ".png", 0) 
		merged_mask = np.zeros(img.shape)

		for mk in os.listdir(masks_path):
			f = cv2.imread(masks_path + mk , 0)
			print(fn ,"    ", mk)
			merged_mask += f

		merged_mask = np.minimum(merged_mask, 255)

		cv2.imwrite(masks_path + "../" + fn + '_mask.png',merged_mask)
