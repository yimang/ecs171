# # ECS 171 - Homework 2
# ## WHERE  DID  THE  BAKER  GO? [100 PT]
# In this exercise, you will build a classifier that can find the localization site of a protein in yeast,based on 8 attributes (features). You will use the “Yeast” dataset (“yeast.data” file; 1484 proteins,8 features, 10 different classes; no missing data) that is available in the UCI Machine LearningRepository:<br /><br />
# https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/

# ## Q1
# Construct a 3-layer artificial neural network (ANN) and specifically a feed-forward multi-layer perceptron to perform multi-class classification.   The hidden layer should have 3 nodes.   Split your data into a random set of 70% of the samples as the training set and the rest 30% as the testing set.  For training, use stochastic gradient descent with back-propagation.  Please note that you will never train with the testing set; the ANN will only take into account the training set for updating the weights.  For the most pupular class "CYT",  provide  2  plots:  (I)  weight  values  per  iteration  for  the  last  layer  (3  weights  and bias), (II) training and test error per iteration. [30pt]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

columns = ["Sequence Name", "mcg", "gvm", "alm", "mit", "erl", "pox", "vac", "nuc", "class"]
classes = ["CYT", "NUC", "MIT", "ME3", "ME2", "ME1", "EXC", "VAC", "POX", "ERL"]

# load and process the data
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data", names=columns, delim_whitespace=True)
df = df.drop("Sequence Name", 1)
# replace classes with a number from 0-9 that corresponds to each class
for i in range(10):   
    df = df.replace(to_replace=classes[i], value=i)
# data is randomized to get a more consistent distribution for training and test data
df_randomized = df.sample(frac=1).reset_index(drop=True)
# 70% data for training set and 30% data for testing set
training_set = df_randomized.loc[0:int(df.shape[0] * 0.7)]
testing_set = df_randomized.loc[int(df.shape[0] * 0.7 + 1):df.shape[0] - 1]

import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

features = ["mcg", "gvm", "alm", "mit", "erl", "pox", "vac", "nuc"]

# prepare training set and test set; one hot encoding for y values
y_train = np.array(pd.DataFrame(training_set["class"]))
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = np.array(pd.DataFrame(testing_set["class"]))
y_test = keras.utils.to_categorical(y_test, num_classes=10)
x_train = np.array(pd.DataFrame(training_set[features]))
x_test = np.array(pd.DataFrame(testing_set[features]))

# callback function to obtain weights and bias for plotting
class My_Callback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.biases = []
        self.w1 = []
        self.w2 = []
        self.w3 = []
 
    def on_epoch_end(self, epoch, logs={}):
        biases = model.layers[2].get_weights()[1][0]
        w1 = model.layers[2].get_weights()[0][0][0]
        w2 = model.layers[2].get_weights()[0][1][0]
        w3 = model.layers[2].get_weights()[0][2][0]
        self.biases.append(biases)
        self.w1.append(w1)
        self.w2.append(w2)
        self.w3.append(w3)

# Define model of neural network
model = Sequential()
model.add(Dense(3, activation='sigmoid', input_dim=8))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

num_epochs = 1000
my_callback = My_Callback()

# train network
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[my_callback], verbose=0, epochs=num_epochs, batch_size=1)    

# For error, it's for the whole data set rather than just CYT
# For weights and biases, it's only for CYT
error = 1 - np.array(history.history['acc'])
error_val = 1 - np.array(history.history['val_acc'])
# plot error
train_acc, = plt.plot(error, label='train')
test_acc, = plt.plot(error_val, label='test')
plt.title('Error')
plt.xlabel('epoch')
plt.ylabel('Error %')
plt.legend(handles=[train_acc, test_acc])
plt.show()
# plot weights and bias for CYT
bias_label, = plt.plot(my_callback.biases, label='bias')
w1_label, = plt.plot(my_callback.w1, label='w1')
w2_label, = plt.plot(my_callback.w2, label='w2')
w3_label, = plt.plot(my_callback.w3, label='w3')
plt.title('Weights and Bias')
plt.xlabel('epoch')
plt.ylabel('value')
plt.legend(handles=[bias_label, w1_label, w2_label, w3_label])
plt.show()

# ## Q2
# Now re-train the ANN with all your data (all 1484 samples). What is your training error? Provide the final activation function formula for class "CYT" after training. [10pt]

# train with all data
y_train_all = np.array(pd.DataFrame(df_randomized["class"]))
y_train_all = keras.utils.to_categorical(y_train_all, num_classes=10)
x_train_all = np.array(pd.DataFrame(df_randomized[features]))

# Define model of neural network
model_2 = Sequential()
model_2.add(Dense(3, activation='sigmoid', input_dim=8))
model_2.add(Dropout(0.5))
model_2.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_2.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

num_epochs_2 = 1000

history = model_2.fit(x_train_all, y_train_all, verbose=1, epochs=num_epochs_2, batch_size=1)    
score = model_2.evaluate(x_train_all, y_train_all, batch_size=y_train_all.shape[0], verbose=0)
print(1 - score[1])


# ## Q3
# For the ANN that you have built (3 layers, 1 hidden layer, 3 hidden nodes) calculate the first round of weight update with back-propagation with paper and pencil for all weights but for only the first sample.  Confirm that the numbers that you calculated are the same with those produced by the code and provide both your calculations and the code out-put.  Provide both calculations made by hand (scanned image is fine) and corresponding output from the program that shows that both are in agreement. [30pt]

# new callback function to store weights and bias before and after one gradient descent update
class My_Callback_Q3(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.l1bias = []
        self.l2bias = []
        self.l1w = []
        self.l2w = []
 
    # store weights before GD
    def on_batch_begin(self, batch, logs={}):
        l1bias = self.model.layers[0].get_weights()[1]
        l1w = self.model.layers[0].get_weights()[0]
        l2bias = self.model.layers[1].get_weights()[1]
        l2w = self.model.layers[1].get_weights()[0]
        self.l1bias.append(l1bias)
        self.l1w.append(l1w)
        self.l2bias.append(l2bias)
        self.l2w.append(l2w)
 
    # store weights after GD
    def on_batch_end(self, batch, logs={}):
        l1bias = self.model.layers[0].get_weights()[1]
        l1w = self.model.layers[0].get_weights()[0]
        l2bias = self.model.layers[1].get_weights()[1]
        l2w = self.model.layers[1].get_weights()[0]
        self.l1bias.append(l1bias)
        self.l1w.append(l1w)
        self.l2bias.append(l2bias)
        self.l2w.append(l2w)
        
x_sample = x_train[0].reshape(1, 8)
y_sample = y_train[0].reshape(1, 10)

# Define model of neural network
model_3 = Sequential()
model_3.add(Dense(3, activation='sigmoid', input_dim=8))
model_3.add(Dense(10, activation='sigmoid'))

sgd = SGD(lr=0.01)
model_3.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

num_epochs = 1
my_callback_Q3 = My_Callback_Q3()

history = model_3.fit(x_sample, y_sample, callbacks=[my_callback_Q3], verbose=1, epochs=num_epochs, batch_size=1)    

# initial values before gradient descent
X = x_sample.T
Y = y_sample.T
b1 = my_callback_Q3.l1bias[0].reshape(3, 1)
b2 = my_callback_Q3.l2bias[0].reshape(10, 1)
w1 = my_callback_Q3.l1w[0].T
w2 = my_callback_Q3.l2w[0].T

# values after gradient descent with keras
b1_after = my_callback_Q3.l1bias[1].reshape(3, 1)
b2_after = my_callback_Q3.l2bias[1].reshape(10, 1)
w1_after = my_callback_Q3.l1w[1].T
w2_after = my_callback_Q3.l2w[1].T

# I will manually execute gradient descent, then compare the difference between my values and keras values

# Forward pass
z1 = np.dot(w1, X) + b1
a1 = 1 / (1 + np.exp(-z1))
z2 = np.dot(w2, a1) + b2
a2 = 1 / (1 + np.exp(-z2))

# Backpropagation
dz2 = a2 - Y
dw2 = np.dot(dz2, a1.T)
db2 = dz2
dz1 = np.dot(w2.T, dz2) * (a1 * (1-a1))
dw1 = np.dot(dz1, X.T)
db1 = dz1

# Gradient descent
w1 = w1 - 0.01 * dw1
w2 = w2 - 0.01 * dw2
b1 = b1 - 0.01 * db1
b2 = b2 - 0.01 * db2

# My values
print(w1)
print(w2)
print(b1)
print(b2)
# keras values
print(w1_after)
print(w2_after)
print(b1_after)
print(b2_after)
# differences
print(w1 - w1_after)
print(w2 - w2_after)
print(b1 - b1_after)
print(b2 - b2_after)

# ## Q4
# Increase the number of hidden layers from 1 to 2 and then to 3. Then increase the number of hidden nodes per layer from 3 to 6,  then to 9 and finally to 12.   Create a 3x4 matrix with the number of hidden layers as rows and the number of hidden nodes per layer as columns, with each element (cell) of the matrix representing the testing set error for that specific combination of layers/nodes.  What is the optimal configuration?  What you find the relationship between these attributes (number of layers, number of nodes) and the generalization error (i.e. error in testing data) to be? [25pt]

# function to train model with flexible parameters
def train_model(num_hidden_layers, num_hidden_nodes, num_epochs, bat_size):
    if (num_hidden_layers < 1):
        print("Please enter a number greater than 1 for the number of hidden layers")
        return null
    
    model = Sequential()
    model.add(Dense(num_hidden_nodes, activation='sigmoid', input_dim=8))
    model.add(Dropout(0.5))
    for i in range(num_hidden_layers - 1):
        model.add(Dense(num_hidden_nodes, activation='sigmoid'))
        model.add(Dropout(0.5))        
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, verbose=0, epochs=num_epochs, batch_size=bat_size) 
    return model

testing_set_error = np.zeros((3, 4))

# loop over number of hidden layers
for i in range(3):
    # loop over number of hidden nodes
    for j in range(4):
        print(str(i+1) + " hidden layers and " + str(3*(j+1)) + " hidden nodes")
        model_4 = train_model(i + 1, 3*(j+1), 1000, 1)
        score_4 = model_4.evaluate(x=x_test, y=y_test, batch_size=y_test.shape[0], verbose=0)
        testing_set_error[i][j] = 1 - score_4[1]

print (testing_set_error)
      
# ## Q5
# Which class does the following sample belong to? [5pt]<br /> Unknown Sample 0.49 0.51 0.52 0.23 0.55 0.03 0.52 0.39

# best performing model was with 1 hidden layer and 12 hidden nodes
model_5 = train_model(1, 12, 1000, 1)
score_5 = model_5.evaluate(x=x_test, y=y_test, batch_size=y_test.shape[0], verbose=0)
print(1 - score_5[1])

input = np.array([[0.49, 0.51, 0.52, 0.23, 0.55, 0.03, 0.52, 0.39]])
model_5.predict(input, 1)


# ## Q6
# Can you come up with a quantitative measure of uncertainty for each classification? What is the uncertainty for the unknown sample of the previous question? Justify your assumptions and method [5pt bonus]

# See report

