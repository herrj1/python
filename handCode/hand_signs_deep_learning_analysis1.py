import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# import warnings
import warnings
# filter warning
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


#load dataset
x_l = np.load('../x.npy')
Y_l = np.load('../Y_npy')

#resize images in dataset
img_size = 64
plt.subplot(1,2,1)
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(x_l[900].reshape(img.size, img_size))
plt.axis('off')


#join arrays
X = np.concatenate((x_l[204:409], x_l[822:1027]), axis=0)
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, 0), axis=0).reshape(X.shape[0], 1)
print("X shape: ", X.shape)
print("Y shape: ", Y.shape)


#create X_train, y_train, X_test, y_test arrays
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.15, random_state=42)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]

X_train_flatten = X_train.reshape(number_of_train, X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test.reshape(number_of_test,X_test.shape[1]*X_test.shape[2])
print("X train flatten", X_train_flatten.shape)
print("X test flatten", X_test_flatten.shape)

X_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T
print("x train: ", x_train.shape)
print("x test: ", x_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)


#example
def dummy(parameter):
    dummy_parameter = parameter + 5
    return dummy_parameter
result = dummy(3)

def initialize_weights_and_bias(dimension):
    w = np.full((dimension, 1), 0.01)
    b = 0.0
    return w, b

def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

y_head = sigmoid(0)
y_head


def forward_propagation(w,b,x_train,y_train):
    z = np.dot(w,T,x_train)+b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    return cost

def forward_backward_propagation(w,b,x_train,y_train):
    #forward propagation
    z = np.dot(w.T,x_train)+b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    #backward propagation
    derivative_weight = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    return cost, gradients

#updating(learning) parameters
def update(w,b,x_train,y_train,learning_rate,number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []

    #updating(learning) parameters is number_of_iteration times
    for i in range(number_of_iteration):
        #make forward and backword propagation and find cost and gradients
        cost, gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)

        #lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" %(i, cost))

        #update(learn) parameters weights and bias
        parameters = {"weight": w, "bias": b}
        plt.plot(index,cost_list2)
        plt.xticks(index,rotation='vertical')
        plt.xlabel("Number of Iteration")
        plt.ylabel("Cost")
        plt.show()
        return parameters, gradients, cost_list

def predict(w,b,x_test):
    #x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    #if z is bigger than 0.5, our prediction is sign one(y_head=1),
    #if z is smaller than 0.5, our prediction is sign zero(y_head=0)
    for i in range(z.shape[1]):
        if z[0,i] <= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction


def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
    #initialize
    dimension = x_train.shape[0]
    w,b = initialize_weights_and_bias(dimension)

    #do not change learning rate
    parameters, gradients, cost_list = update(w,b,x_train,y_train,learning_rate,num_iterations)

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)

    #print train/test errors
    print("Train accuracy: {} %".format(100 - np.mean(np.abs(y_predict_train - y_train))*100))
    print("Test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test))*100))


logistic_regression(x_train,y_train,x_test,y_test,learning_rate = 0.01, num_iterations = 150)


from sklearn import linear_model
logreg = linear_model.LogisticRegression(random_state = 42,max_iter = 150)
print("Test accuracy: {}".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))
print("Train accuracy: {}").format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T))

