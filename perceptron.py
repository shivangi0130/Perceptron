# M=0, R=1

import numpy
import pandas
from sklearn.utils import shuffle

# load data set
# data format x1 x2 x3 x4 ... y
sonar_data = pandas.read_csv('A:\Dataset\sonar.all-data.csv', header=None)

# shuffle the data
sonar_data = shuffle(sonar_data)

rows = sonar_data.shape[0]
cols = sonar_data.shape[1]

# X is the set of input features and Y is the output
X = sonar_data.iloc[:, 0:cols - 1]  # input featurs x1 x2 x3 x4 ... xn
Y = sonar_data.iloc[:, cols - 1]
Y = Y.map({'M': 0, 'R': 1})
Y = Y.reset_index(drop=False)
del Y['index']

# weights/coefficients
no_of_weights = X.shape[1]
# X_weights=numpy.random.rand(no_of_weights+1)
X_weights = numpy.zeros(no_of_weights + 1)
k = 3
# splitting data into training and testing sets
epoch_size = rows / k
learning_rate = 0.1

for epoch in range(1, 500):
    sum_error = 0

    for i in range(0, 50):
        # activation
        #        print('i=',i)
        activation = X_weights[0] + sum(X_weights[1:] * (X.iloc[i, :]))
        #        print('activation=',activation)
        if activation >= 0:
            prediction = 1
        else:
            prediction = 0
        expected = Y[60][i]
        #        print('expected=',expected)
        error = expected - prediction
        #        print('error=',error)
        sum_error = sum_error + (error ** 2)
        X_weights[0] = X_weights[0] + learning_rate * error
        for j in range(0, no_of_weights):
            X_weights[j] = X_weights[j] + learning_rate * error * X.iloc[i, j]
    print('epoch', epoch, 'error', sum_error)