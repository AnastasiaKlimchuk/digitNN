from utils import read_data, split_test, accuracy, write_data
import Model
import pandas as pd
import numpy as np

### CONSTANTS DEFINING THE MODEL ####
n_x = 784     # num_px * num_px * 3
n_h1 = 16
n_h2 = 16
n_y = 10
layers_dims = (n_x, n_h1, n_h2, n_y)

# Extract Train and Test data by solution
labels, X_train, X_test = read_data('train.csv', 'test.csv')

# Give train and test data from X_train and labels
train_set_x, train_set_y, test_set_x, test_set_y = split_test(X_train, labels)
# standardize the data
train_set_x = train_set_x / 255
test_set_x = test_set_x / 255

test = X_test.T
test = test / 255

parameters = Model.L_layer_model(train_set_x, train_set_y, layers_dims, learning_rate=0.9, num_iterations=12000, print_cost=True)


pred_train = Model.predict(train_set_x, parameters)
print('train set accuracy: ', accuracy(pred_train, train_set_y))

pred_test = Model.predict(test_set_x, parameters)
print('test set accuracy: ', accuracy(pred_test, test_set_y))

result = Model.predict(test, parameters)

write_data(result)





