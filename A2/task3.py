from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

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

image_size = 28
num_labels = 10
num_h1_units = 1000
batch_size = 128


def run_NN(file, num_h1_units=1000, learning_rate=0.01, momentum=0.0):

	# Making the model
	graph = tf.Graph()
	with graph.as_default():
		X = tf.placeholder(tf.float32, shape=(None, image_size * image_size))
	 	Y = tf.placeholder(tf.float32, shape=(None, num_labels))

		# between X and hidden
		layer1_weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_h1_units]))
		layer1_biases = tf.Variable(tf.zeros([num_h1_units]))
		# between hidden and output
		layer2_weights = tf.Variable(tf.zeros([num_h1_units, num_labels]))
		layer2_biases = tf.Variable(tf.zeros([num_labels]))

		hidden = tf.nn.relu(tf.matmul(X, layer1_weights) + layer1_biases)
		logits = tf.matmul(hidden, layer2_weights) + layer2_biases

		train_prediction = tf.nn.softmax(logits)
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, Y))
		loss = cross_entropy

		optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss) # Use loss to add regularization

		correct_prediction = tf.equal(tf.argmax(train_prediction, 1), tf.argmax(Y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


	# Running the model


	with tf.Session(graph=graph) as session:
		tf.initialize_all_variables().run()
		for epoch in range(200):
			for i in range(150):
				batch_data = training_set[i*100 : (i+1)*100]
				batch_labels = training_labels[i*100 : (i+1)*100]

				feed_dict = {X : batch_data, Y : batch_labels}

				_, ce = session.run(
				  [optimizer, cross_entropy], feed_dict=feed_dict)

			if epoch % 1 == 0:
				cost_train, accuracy_train = session.run([cross_entropy, accuracy],
					feed_dict={X: training_set, Y: training_labels})
				cost_eval, accuracy_eval = session.run([cross_entropy, accuracy],
					feed_dict={X: validation_set, Y: validation_labels})
				cost_test, accuracy_test = session.run([cross_entropy, accuracy],
					feed_dict={X: testing_set, Y: testing_labels})
				errors_train = training_set.shape[0]*(1-accuracy_train)
				errors_eval = validation_set.shape[0]*(1-accuracy_eval)
				errors_test = testing_set.shape[0]*(1-accuracy_test)
				print ("Epoch:%04d, t_cost=%0.9f, t_acc=%0.4f, t_err=%0.4f, v_cost=%0.9f, v_acc=%0.4f, v_err=%0.4f, te_acc=%0.4f, te_err=%0.4f " %
					(epoch, cost_train, accuracy_train, errors_train, cost_eval, accuracy_eval, errors_eval, accuracy_test, errors_test), file=file)

fout = open("task3_results.log", 'w', 0)
#fout = sys.stdout
for num_hidden in [100, 500]:
	for learning_rate in [0.001]:
		print ("Starting experiment with number of hidden units {} and learning_rate {}".format(num_hidden, learning_rate), file=fout)
		run_NN(fout, num_h1_units=num_hidden, learning_rate=learning_rate)
fout.close()