from __future__ import print_function
import numpy as np
import tensorflow as tf
from utils import *

# laod data
data = np.load("data100D.npy")
data = np.load("data2D.npy")
data = np.array([[-1, -1],
				 [0, 1],
				 [1,-1],
				 [0, 0]])
print ("shape of input data: {}".format(data.shape))
num_samples = data.shape[0]
data_dimension = data.shape[1]

sqrt_pi_inv = 1 / np.sqrt(np.pi)

# Task 2.1.2
def log_prob_px_given_z(samples, means, stdevs):
	# (num_samples, dimesion) -> (1, num_samples, dimension)
	expanded_samples = tf.expand_dims(samples, 0) 
	# (k_clusters, dimension) -> (k_clusters, 1, dimension)
	expanded_means = tf.expand_dims(means, 1)
	# (k_clusters, num_samples)
	numerator = tf.reduce_sum(tf.square(tf.sub(expanded_samples, expanded_means)), 2)
	denom_inv = tf.inv(2*tf.square(stdevs))
	prob = sqrt_pi_inv * tf.sqrt(denom_inv) * tf.exp(-numerator * denom_inv)
	# (k_clusters, num_samples)
	return tf.log(prob), numerator, denom_inv

def log_prob_pz_given_x(log_px_given_z, pis):
	# probs is tensor given by log_prob_px_given_z()
	# pis is the apriori probability of each cluster
	numerator = pis * tf.exp(log_px_given_z)
	denom_inv = tf.inv(reduce_logsumexp(pis * log_px_given_z, reduction_indices=0, keep_dims=True))
	return numerator * denom_inv, numerator, tf.inv(denom_inv)

k_clusters = 3
gaussian_norm_coef = 1 / np.sqrt(2*np.pi)

graph = tf.Graph()
with graph.as_default():
	input_data = tf.placeholder(tf.float32, shape=(None, data_dimension))
	pis = tf.Variable(tf.random_uniform([k_clusters, 1], minval=0, maxval=1))
	pis = pis / tf.reduce_sum(pis)
	means = tf.Variable(tf.truncated_normal([k_clusters, data_dimension]))
	stdevs = tf.Variable(tf.truncated_normal([k_clusters, 1], mean=1.0, stddev=0.1))
	log_px_given_z, a, b = log_prob_px_given_z(input_data, means, stdevs)
	log_pz_given_x, c, d = log_prob_pz_given_x(log_px_given_z, pis)

with tf.Session(graph=graph) as session:
	tf.initialize_all_variables().run()
	log_px_given_z_value = session.run(log_px_given_z, feed_dict={input_data: data})
	# log_pz_given_x_value = session.run(log_pz_given_x, feed_dict={input_data: data})

	print (a.eval(feed_dict={input_data: data}))
	print (b.eval(feed_dict={input_data: data}))
	print ("\n\n")
	# print (means.eval())
	# print (stdevs.eval())
	print (log_px_given_z_value)
	print (pis.eval())
	print ("\n\n")

	print (np.exp(log_px_given_z_value))
	print (c.eval(feed_dict={input_data: data}))
	print (d.eval(feed_dict={input_data: data}))