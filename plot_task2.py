import numpy as np
import matplotlib.pyplot as plt
import sys
import re

input_file = sys.argv[1]

line_count = 0
data = []

with open(input_file, 'r') as infile:
	for line in infile:
		if line_count == 0:
			line_count += 1
			continue
		if line[0:8] == "Starting":
			data = np.array(data)
			data = data[0:10]
			epochs = data[:,0]
			training_cross_entropy = data[:,1]
			training_errs = data[:,3]
			validation_cross_entropy = data[:,4]
			validation_errs = data[:,6]
			# Two subplots, the axes array is 1-d
			f, axarr = plt.subplots(2, sharex=True)
			axarr[0].plot(epochs, -1*training_cross_entropy, epochs, -1*validation_cross_entropy)
			axarr[0].set_title('Cross Entropy')

			axarr[1].plot(epochs, training_errs, epochs, validation_errs)
			axarr[1].set_title('Errors')
			plt.show()
			data = []
		else:
			data.append([float(i) for i in re.findall(r'\d+\.\d+|\d+', line)])
		line_count += 1
