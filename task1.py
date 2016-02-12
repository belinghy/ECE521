from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load data
with np.load("notMNIST.npz") as data:
	images, labels = data["images"], data["labels"]

# Convert labels and data to usable format
labels_temp = []
for label in labels:
	temp = np.zeros(10)
	temp[label[0]] = 1
	labels_temp.append(temp)

labels = np.array(labels_temp)
images = images / 255.0
images = images.reshape(28*28, 18720).T

training_set = images[0:15000]
training_labels = labels[0:15000]
validation_set = images[15000:16000]
validation_labels = labels[15000:16000]
testing_set = images[16000:]
testing_labels = labels[16000:]

log = open("task1_results.log", "a")

for num_epochs in range(1000, 2000, 1000):
	print ("Training with epochs: {}".format(num_epochs))
	with tf.Session() as sess:
		X = tf.placeholder(tf.float32, shape=[None, 28*28])
		Y = tf.placeholder(tf.float32, shape=[None, 10])

		W = tf.Variable(tf.zeros([28*28, 10]))
		b = tf.Variable(tf.zeros([10]))

		prediction = tf.nn.softmax(tf.matmul(X, W) + b)
		cross_entropy = -tf.reduce_sum(Y * tf.log(prediction))

		train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(cross_entropy)
		init = tf.initialize_all_variables()
		sess.run(init)

		num_of_input = training_set.shape[0]
		batch_size = 1
		for i in range(num_epochs):
			random_data_point = np.random.randint(num_of_input, size=batch_size)
			batch_xs = training_set[random_data_point]
			batch_labels = training_labels[random_data_point]
			sess.run(train_step, feed_dict={X: batch_xs, Y: batch_labels})

		correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		training_errors = training_set.shape[0] * \
			(1.0 - accuracy.eval(feed_dict={X: training_set, Y: training_labels}))
		training_log_likelihood = -1 * cross_entropy.eval(feed_dict={X: training_set, Y: training_labels})
		validation_errors = validation_set.shape[0] * \
			(1.0 - accuracy.eval(feed_dict={X: validation_set, Y: validation_labels}))
		validation_log_likelihood = -1 * cross_entropy.eval(feed_dict={X: validation_set, Y: validation_labels})
		"""
		print("{:10} {:10.5f} {:10.5f} {:10.5f} {:10.5f}".format(
				num_epochs,
				training_errors,
				training_log_likelihood,
				validation_errors,
				validation_log_likelihood),
				file = log)
		"""

log.close()