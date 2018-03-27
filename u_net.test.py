import u_net
import tensorflow as tf
import numpy as np
import cv2
import random
import os

u = u_net.u_net("gpu:2", 0, batch_size=1, output_dim=1)
logits = tf.nn.sigmoid(u.outputs, name="pred")

saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())
saver.restore(sess,tf.train.latest_checkpoint('models/'))

DATA_PATH = "data/test/"
for fn in os.listdir(DATA_PATH):
	img = cv2.imread(DATA_PATH + fn + "/images/" + fn + ".png")
	img_shape = img.shape
	p = sess.run(logits, feed_dict={u.inputs: [cv2.resize(img, (572, 572))]})
	mask = p[0]
	print(mask.shape,"   ", img_shape)
	mask = cv2.resize(mask,(img_shape[1],img_shape[0]))
	print(mask.shape,"\n")

	t = 0.5
	mask = mask * (mask > t)
	mask[mask>t] = 255
	mask = mask.astype(int)

	cv2.imwrite(DATA_PATH + fn + "/images/" + fn + "_predicted_mask.png", mask)
