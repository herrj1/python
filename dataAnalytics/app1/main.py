import numpy as np
import matplotlib.pyplot as plt
import csv

#Load the dataset and parse it
def load_dataset():
	num_rows = sum(1 for line in open('Facebook_metrics\\dataset_Facebook.csv')) - 1
	X = np.zeros((num_rows, 1))
	y = np.zeros((num_rows, 1))
	with open('Facebook_metrics\\dataset_Facebook.csv') as f:
		reader = csv.DictReader(f, delimiter=';')
		next(reader, None)
		for i, row in enumerate(reader):
			X[i] = int(row['share']) if len(row['share']) > 0 else 0
			y[i] = int(row['like']) if len(row['like']) > 0 else 0
			
	return X, y

#Visualize dataset
def visualize_dataset(X, y):
	plt.xlabel('Number of shares')
	plt.ylabel('Number of likes')
	plt.scatter(X, y)
	plt.show()

#Main driver
if __name__ == '__main__':
	X, y = load_dataset()
	visualize_dataset(X, y)