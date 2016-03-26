from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *

# laod data
data = np.load("data100D.npy")
# data = np.load("data2D.npy")
"""
data = np.random.multivariate_normal([0, 0], [[0.01,0], [0,0.01]], size=100)
data = np.append(data, np.random.multivariate_normal([1, 1], [[0.01,0], [0,0.01]], size=100), 0)
data = np.append(data, np.random.multivariate_normal([-1, -1], [[0.01,0], [0,0.01]], size=100), 0)
"""

print ("shape of input data: {}".format(data.shape))
num_samples = data.shape[0]
data_dimension = data.shape[1]

sqrt_pi_inv = 1 / np.sqrt(np.pi)

# Task 2.1.2
def log_prob_px_given_z(samples, means, variances):
	# (num_samples, dimesion) -> (1, num_samples, dimension)
	expanded_samples = tf.expand_dims(samples, 0) 
	# (k_clusters, dimension) -> (k_clusters, 1, dimension)
	expanded_means = tf.expand_dims(means, 1)
	# (k_clusters, num_samples)
	numerator = tf.reduce_sum(tf.square(tf.sub(expanded_samples, expanded_means)), 2)
	denom_inv = tf.inv(2*variances)
	prob = sqrt_pi_inv * tf.sqrt(denom_inv) * tf.clip_by_value(tf.exp(-numerator * denom_inv), 1e-5, 1e5)
	# (k_clusters, num_samples)
	return tf.log(prob), numerator, denom_inv

def log_prob_pz_given_x(log_px_given_z, pis):
	# probs is tensor given by log_prob_px_given_z()
	# pis is the apriori probability of each cluster
	marginal = pis * tf.exp(log_px_given_z)
	log_marginal = tf.log(marginal)
	denom = reduce_logsumexp(log_marginal, reduction_indices=0, keep_dims=True)
	return log_marginal - denom, log_marginal, denom

k_clusters = 3
learning_rate=0.1
beta1=0.9
beta2=0.99
epsilon=1e-5
gaussian_norm_coef = 1 / np.sqrt(2*np.pi)

graph = tf.Graph()
with graph.as_default():
	input_data = tf.placeholder(tf.float32, shape=(None, data_dimension))
	pis_temp = tf.Variable(tf.truncated_normal([k_clusters, 1]))
	pis = tf.exp(logsoftmax(pis_temp))
	means = tf.Variable(tf.truncated_normal([k_clusters, data_dimension]))
	variances_temp = tf.Variable(tf.truncated_normal([k_clusters, 1]))
	variances = tf.exp(variances_temp)
	
	log_px_given_z, a, b = log_prob_px_given_z(input_data, means, variances)
	marginal_pxz = tf.log(pis) + log_px_given_z
	loss = -tf.reduce_sum(reduce_logsumexp(marginal_pxz))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon).minimize(loss)
	# log_pz_given_x, c, d = log_prob_pz_given_x(log_px_given_z, pis)

with tf.Session(graph=graph) as session:
	tf.initialize_all_variables().run()
	for iteration in range(1000):
		# log_px_given_z_value = session.run(log_px_given_z, feed_dict={input_data: data})
		_, loss_value = session.run([optimizer, loss], feed_dict={input_data: data})
		print ("{:4d}, {:0.5f}".format(iteration+1, loss_value))
		# print (np.exp(log_px_given_z.eval(feed_dict={input_data: data})))
		print (pis.eval(feed_dict={input_data: data}))
		#print (marginal_pxz.eval(feed_dict={input_data: data}))
		#print (px.eval(feed_dict={input_data: data}))

	means_value = means.eval()
	plt.scatter(data[:,0], data[:,1])
	for mean in means_value:
		# print (mean)
		plt.plot(mean[0], mean[1], markersize=35, marker="x", color="green", mew=10)
	plt.show()

	"""
	log_pz_given_x_value = session.run(log_pz_given_x, feed_dict={input_data: data})

	print (a.eval(feed_dict={input_data: data}))
	print (b.eval(feed_dict={input_data: data}))
	print ("\n\n")
	# print (means.eval())
	# print (variances.eval())
	print (log_px_given_z_value)
	print (pis.eval())
	print ("\n\n")

	print (log_pz_given_x_value)
	print (c.eval(feed_dict={input_data: data}))
	print (d.eval(feed_dict={input_data: data}))
	"""