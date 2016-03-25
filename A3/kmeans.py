from __future__ import print_function
import numpy as np
import tensorflow as tf

# load data
data = np.load("data2D.npy")
print ("shape of input data: {}".format(data.shape))
num_samples = data.shape[0]
data_dimension = data.shape[1]



# Task 1.1.2
def euclidean_distance(samples, centroids):
	# (num_samples, dimesion) -> (1, num_samples, dimension)
	expanded_samples = tf.expand_dims(samples, 0) 
	# (k_clusters, dimension) -> (k_clusters, 1, dimension)
	expanded_centroids = tf.expand_dims(centroids, 1)
	# (k_clusters, num_samples)
	distances = tf.reduce_sum(tf.square(tf.sub(expanded_samples, expanded_centroids)), 2)
	return distances


# Parameters
k_clusters = 3
learning_rate = 0.1
beta1 = 0.9
beta2 = 0.99
epsilon = 1e-5

graph = tf.Graph()
with graph.as_default():
	# Tensorflow placeholders and variables
	input_data = tf.placeholder(tf.float32, shape=(num_samples, data_dimension))
	centroids = tf.Variable(tf.truncated_normal([k_clusters, data_dimension]))

	# Add up all the distances
	distances = euclidean_distance(input_data, centroids)
	# nearest_clusters = tf.to_int32(tf.argmin(distances, 0))
	min_distances = tf.reduce_min(distances, 0)
	loss = tf.reduce_sum(min_distances)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon).minimize(loss)


with tf.Session(graph=graph) as session:
	# initialize centroids
	tf.initialize_all_variables().run()
	for iteration in range(500):
		_, loss_value = session.run([optimizer, loss], feed_dict={input_data: data})
		print (loss_value)
