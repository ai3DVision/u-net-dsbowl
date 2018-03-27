# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
from u_net import u_net, total_loss

if __name__ == "__main__":
	import time
	import tensorflow as tf

	DEVICE_STR = "gpu:1"
	BATCH_SIZE = 4
	OUTPUT_DIM = 1
	epsilon = 1e-5

	with tf.device(DEVICE_STR):
		u = u_net(DEVICE_STR, 1e-5, BATCH_SIZE, OUTPUT_DIM)
		output_shape = u.outputs.get_shape().as_list()
		logits = tf.nn.sigmoid(u.outputs, name="pred")

		y_placeholder = tf.placeholder(dtype=tf.float32, shape=output_shape)

		loss =  - tf.reduce_mean(y_placeholder * tf.log(logits + epsilon) + (1 - y_placeholder) * tf.log(1 - logits + epsilon))

		loss, _, reloss = total_loss("loss", loss)

		opt = tf.train.AdamOptimizer(learning_rate=1e-6)

		update_op = opt.minimize(loss)

	tf.summary.scalar("loss", loss)
	tf.summary.scalar("re loss", reloss)
	tf.summary.image("inputs", u.inputs)
	tf.summary.image("labels", y_placeholder)
	tf.summary.image("outputs", logits)

	merged = tf.summary.merge_all()

	DATA_PATH = "data/train/"
	images = []
	labels = []

	for fn in os.listdir(DATA_PATH):
		img = cv2.imread(DATA_PATH + fn + "/images/" + fn + ".png")
		images.append(cv2.resize(img, (572, 572)))
		mask = cv2.imread(DATA_PATH + fn + "/" + fn + "_mask.png", 0)
		mask = mask / 255
		labels.append(np.reshape(cv2.resize(mask, (572, 572)), (572, 572, 1)))
	
	images = np.array(images)
	labels = np.array(labels)

	data_len = len(images)

	i = 0

	saver = tf.train.Saver()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		train_writer = tf.summary.FileWriter('log/u_net.T1', sess.graph)
		model_file = tf.train.latest_checkpoint("models")
		saver.restore(sess, model_file)
		#sess.run(tf.global_variables_initializer())
		while True:
			image_batch = []
			label_batch = []
			for ix in np.random.randint(data_len, size=(BATCH_SIZE,)):
				image_batch.append(images[ix])
				label_batch.append(labels[ix])
			image_batch = np.array(image_batch)
			label_batch = np.array(label_batch)

			feed_dict = {
				u.inputs: image_batch,
				y_placeholder: label_batch
			}
			summary, l, rl, _ = sess.run([merged, loss, reloss, update_op], feed_dict=feed_dict)
			if i > 0 and i % 1000 == 0:
				saver.save(sess, 'models/u-net.%d'%i, global_step=i)
			if i % 10 == 0:
				train_writer.add_summary(summary, i)
				print("<%d> loss: %.6f, re loss: %.6f"%(i, l, rl))
			i += 1

