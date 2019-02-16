#Import Library of Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB
import numpy as np

#assigning predictor and target variables
x= np.array([[-3,2],[1,7], [1,2], [-2,0], [2,3], [-3,0], [-1,1], [1,1], [-1,2], [2,7], [-4,1], [-2,9]])

y = np.array([3, 3, 6, 3, 4, 3, 3, 5, 3, 4, 9, 4])

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(x, y)

#Predict Output 
predicted= model.predict([[1,2],[6,4]])

#Show Results
print predicted