from __future__ import print_function
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt

# load data
data = np.load("data2D.npy")
print ("shape of input data: {}".format(data.shape))
num_samples = data.shape[0]
data_dimension = data.shape[1]

# hold 1/3 of data for validation
rng = np.random.seed(700)
random_samples = np.random.choice(num_samples, num_samples/3, replace=False)
validation_data = data[random_samples]
train_data = data[np.setdiff1d(np.arange(num_samples), random_samples)]


# Task 1.1.2
def euclidean_distance(samples, centroids):
	# (num_samples, dimesion) -> (1, num_samples, dimension)
	expanded_samples = tf.expand_dims(samples, 0) 
	# (k_clusters, dimension) -> (k_clusters, 1, dimension)
	expanded_centroids = tf.expand_dims(centroids, 1)
	# (k_clusters, num_samples)
	distances = tf.reduce_sum(tf.square(tf.sub(expanded_samples, expanded_centroids)), 2)
	return distances


def run_cluster(k_clusters, file=sys.stdout, learning_rate=0.1, beta1=0.9, beta2=0.99, epsilon=1e-5):
	graph = tf.Graph()
	with graph.as_default():
		# Tensorflow placeholders and variables
		input_data = tf.placeholder(tf.float32, shape=(None, data_dimension))
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
		for iteration in range(200):
			_, loss_value = session.run([optimizer, loss], feed_dict={input_data: train_data})
			print ("{:4d}, {:0.5f}".format(iteration+1, loss_value), file=file)
		validation_loss = session.run(loss, feed_dict={input_data: validation_data})
		print ("validation loss: {}".format(validation_loss), file=file)

		distance_values = distances.eval(feed_dict={input_data: train_data})
		mins = np.argmin(distance_values, axis=0)
		print (np.bincount(mins))

		centroid_values = centroids.eval()
		colour = plt.cm.rainbow(np.linspace(0,1, len(centroid_values)))

		plt.scatter(train_data[:,0], train_data[:,1], c=colour[mins])
		for i, centroid in enumerate(centroid_values):
			plt.plot(centroid[0], centroid[1], markersize=35, marker="x", color='green', mew=10)
			print (centroid, file=file)
		plt.show()
		
		

# fout = open("task1_4_results.log", 'w', 0)
fout = sys.stdout
for k_cluster in [1, 2, 3, 4, 5]:
	print ("Starting experiment with k_cluster = {}".format(k_cluster), file=fout)
	run_cluster(k_clusters=k_cluster, file=fout, learning_rate=0.1)
# fout.close()