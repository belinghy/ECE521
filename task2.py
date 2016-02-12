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

image_size = 28
num_labels = 10
num_h1_units = 1000
batch_size = 128

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

# Making the model
graph = tf.Graph()
with graph.as_default():
	tf_train_dataset = tf.placeholder(tf.float32, shape=(None, image_size * image_size))
 	tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))

	# between X and hidden
	layer1_weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_h1_units]))
	layer1_biases = tf.Variable(tf.zeros([num_h1_units]))
	# between hidden and output
	layer2_weights = tf.Variable(tf.truncated_normal([num_h1_units, num_labels]))
	layer2_biases = tf.Variable(tf.zeros([num_labels]))

	def model(data):
		hidden = tf.nn.relu(tf.matmul(tf_train_dataset, layer1_weights) + layer1_biases)
		return tf.matmul(hidden, layer2_weights) + layer2_biases

	logits = model(tf_train_dataset)
	train_prediction = tf.nn.softmax(logits)
	# cross_entropy = -tf.reduce_sum(tf_train_labels * tf.log(train_prediction)) # Original cross entropy
	cross_entropy = -tf.reduce_sum(tf_train_labels * tf.log(tf.clip_by_value(train_prediction, 1e-10, 1.0))) # Hacked cross_entropy to get no NaN
	loss = cross_entropy + tf.reduce_sum(tf.square(layer2_weights))
	#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

	# optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy) # no regularization
	optimizer = tf.train.GradientDescentOptimizer(0.0005).minimize(loss) # Use loss to add regularization
	
	correct_prediction = tf.equal(tf.argmax(train_prediction, 1), tf.argmax(tf_train_labels, 1))
	accuracy_ = 100.0 * tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Running the model
for num_epochs in [10000]:
	with tf.Session(graph=graph) as session:
		tf.initialize_all_variables().run()
		for step in range(num_epochs):
			random_data_point = np.random.randint(training_set.shape[0], size=batch_size)
			batch_data = training_set[random_data_point]
			batch_labels = training_labels[random_data_point]

			feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}

			_, ce, predictions = session.run(
			  [optimizer, cross_entropy, train_prediction], feed_dict=feed_dict)
			
			if (step % 100 == 0):
				#print('{}'.format(l1w))
				print('Minibatch cross_entropy at step %d: %f' % (step, ce))
				print('Minibatch accuracy: %.1f%%' % accuracy_.eval(feed_dict))
				print('Validation accuracy: %.1f%%' % accuracy_.eval({tf_train_dataset : validation_set, tf_train_labels : validation_labels}))
				print('Test accuracy: %.1f%%' % accuracy_.eval({tf_train_dataset : testing_set, tf_train_labels : testing_labels}))