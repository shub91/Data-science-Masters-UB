import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import time
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sn
import pandas as pd


def preprocess():
    """
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    # HINT: Do not forget to add the bias term to your input data
    X = np.hstack((np.ones((n_data,1)),train_data))
    W = initialWeights.reshape((n_features + 1),1)
    thetan = sigmoid(np.dot(X,W))
    err = ((labeli * np.log(thetan)) + ((1 - labeli) * np.log(1 - thetan)))
    error = (-1 * np.sum(err)) / n_data
    error_grad = (thetan - labeli) * X
    error_grad = np.sum(error_grad, axis = 0) / n_data
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))
    N = data.shape[0]
    # M = data.shape[1]
    X = np.hstack((np.ones((N,1)),data))
    Pr = sigmoid(np.dot(X, W)) #posterior probability
    label = np.argmax(Pr, axis = 1)   #max for classes
    label = label.reshape((N, 1))
    # HINT: Do not forget to add the bias term to your input data
    return label

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))
    # HINT: Do not forget to add the bias term to your input data
    net_val = np.hstack((np.ones((n_data,1)),train_data))
    Q = np.zeros((n_data,n_class))
    weight_val = params.reshape(n_feature+1,n_class)
    ## Implementing the Softmax Function: -
    for i in range(n_class):
        Q[:,i] = np.exp(np.dot(weight_val.T[i,:],net_val.T))
    val_sum = np.sum(Q,1)
    for i in range(n_class):
        Q[:,i] = np.divide(Q[:,i],val_sum)
    ## Taking the log
    Q_log = np.log(Q)
    ## Calculating the error value
    error = 0
    for i in range(n_data):
        for j in range(n_class):
            error += labeli[i][j]*Q_log[i][j]
    error *= -1.0/n_data
    Q_newval = Q - labeli
    error_grad = np.dot(Q_newval.T,net_val)/float(n_data)
    error_grad = error_grad.T.flatten()
    return error, error_grad



def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))
    # HINT: Do not forget to add the bias term to your input data
    ## Looking at the shape of rows for n_data and shape of columns of n_feature (as we did in mlrObjFunction)
    N = data.shape[0]
    M = data.shape[1]
    W = W.reshape(M+1,n_class)
    x = np.hstack((np.ones((N,1)),data))
    Q = np.zeros((N,n_class))
    for i in range(n_class):
        Q[:,i] = np.exp(np.dot(W.T[i,:],x.T))
    sum = np.sum(Q,1)
    for i in range(n_class):
        Q[:,i] = np.divide(Q[:,i],sum)
    label = np.argmax(Q,axis = 1)
    label = label.reshape((N,1))
    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label_train = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_train == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_validation = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_validation == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_test = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_test == test_label).astype(float))) + '%')

# creating a confusion matrix
train_conf = confusion_matrix(train_label, predicted_label_train)
test_conf = confusion_matrix(test_label, predicted_label_test)
print(train_conf)
print(test_conf)

# For Creating Classifcation Report for BLR
print(classification_report(test_label, predicted_label_test, labels = list(range(0, 10))))
ans = []
for i in range(10):
    ans.append(np.mean(predicted_label_test[test_label == i] == test_label[test_label == i]))
ans
print(pd.DataFrame({'accuracy': ans}))

trainval = train_conf
trainconf = pd.DataFrame(trainval, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
plt.figure(figsize = (10,7))
sn.set(font_scale=1)
ax = plt.axes()
sn.heatmap(trainconf, annot=True,fmt="d")
ax.set_title('Confusion matrix for Train set')
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')

testval = test_conf
testconf = pd.DataFrame(testval, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
plt.figure(figsize = (10,7))
sn.set(font_scale=1)
ax = plt.axes()
sn.heatmap(testconf, annot=True,fmt="d")
ax.set_title('Confusion matrix for Test set')
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')

train_label_flat = train_label.ravel()
validation_label_flat = validation_label.ravel()
test_label_flat = test_label.ravel()
train_data_new = train_data

# Comment the next 3 lines to train with full dataset
from sklearn.model_selection import train_test_split
train_data_new, _, train_label_flat, _ = train_test_split(train_data, train_label, train_size = 0.2, random_state = 42)
train_label_flat = train_label_flat.ravel()


pfile = 'demo' if len(train_data_new) == 10000 else 'full'
pfile += '.pickle'
pdump = {
    'linear': {},
    'rbf': {},
    'rbf_g1': {}
}
print('Will save the readings to:', pfile)

print("time:", time.ctime())


print('\n-----------Linear kernel-----------\n')
svm_classifier = SVC(kernel = 'linear')
svm_classifier.fit(train_data_new, train_label_flat.ravel())

accu_score = svm_classifier.score(train_data_new, train_label_flat) * 100
print('Training accuracy:' + str(accu_score) + ' %')
pdump['linear']['train'] = accu_score

accu_score = svm_classifier.score(validation_data, validation_label_flat) * 100
print('Validation accuracy:' + str(accu_score) + ' %')
pdump['linear']['valid'] = accu_score

accu_score = svm_classifier.score(test_data, test_label_flat) * 100
print('Test accuracy:' + str(accu_score) + ' %')
pdump['linear']['test'] = accu_score

print("time:", time.ctime())



print('\n-----------RBF Kernel with Gamma: 1-----------\n')
svm_classifier = SVC(kernel = 'rbf', gamma = 1.0)
svm_classifier.fit(train_data_new, train_label_flat.ravel())

accu_score = svm_classifier.score(train_data_new, train_label_flat) * 100
print('Training accuracy:' + str(accu_score) + ' %')
pdump['rbf_g1']['train'] = accu_score

accu_score = svm_classifier.score(validation_data, validation_label_flat) * 100
print('Validation accuracy:' + str(accu_score) + ' %')
pdump['rbf_g1']['valid'] = accu_score

accu_score = svm_classifier.score(test_data, test_label_flat) * 100
print('Test accuracy:' + str(accu_score) + ' %')
pdump['rbf_g1']['test'] = accu_score

print("time:", time.ctime())



print('\n-----------RBF Kernel with default Gamma-----------\n')
svm_classifier = SVC(kernel = 'rbf', gamma = 'auto')
svm_classifier.fit(train_data_new, train_label_flat.ravel())

accu_score = svm_classifier.score(train_data_new, train_label_flat) * 100
print('Training accuracy:' + str(accu_score) + ' %')
pdump['rbf']['train'] = accu_score

accu_score = svm_classifier.score(validation_data, validation_label_flat) * 100
print('\n Validation accuracy:' + str(accu_score) + ' %')
pdump['rbf']['valid'] = accu_score

accu_score = svm_classifier.score(test_data, test_label_flat) * 100
print('\n Test accuracy:' + str(accu_score) + ' %')
pdump['rbf']['test'] = accu_score

print("time:", time.ctime())



print('\n-----------RBF kernel with default Gamma and C: 1 to 100)-----------\n')
c_train_acc = np.zeros(11)
c_test_acc = np.zeros(11)
c_valid_acc = np.zeros(11)

C = [1] + list(range(10, 101, 10))

for i in range(11):
    svm_classifier = SVC(kernel = 'rbf', gamma = 'auto', C = C[i])
    svm_classifier.fit(train_data_new, train_label_flat)

    print('C-Value: ' + str(C[i]))

    c_train_acc[i] = svm_classifier.score(train_data_new, train_label_flat) * 100
    print('Training accuracy: ' + str(c_train_acc[i]) + ' %')

    c_valid_acc[i] = svm_classifier.score(validation_data, validation_label_flat) * 100
    print('Validation accuracy: ' + str(c_valid_acc[i]) + ' %')

    c_test_acc[i] = svm_classifier.score(test_data, test_label_flat) * 100
    print('Test accuracy: ' + str(c_test_acc[i]) + ' %')

print("time:", time.ctime())


pdump['rbf']['c_train'] = c_train_acc
pdump['rbf']['c_valid'] = c_valid_acc
pdump['rbf']['c_test'] = c_test_acc

with open(pfile, 'wb') as pfp:
    pickle.dump(pdump, pfp)
    print('Saved the readings to:', pfile)


"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_train = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_train == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_validation = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_validation == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_test = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_test == test_label).astype(float))) + '%')

readings = pdump
plt.figure(num = None, figsize = (12, 9), dpi = 80, facecolor = 'w', edgecolor = 'k')
x = [1] + list(np.arange(10, 101, 10))
plt.plot(x, readings['rbf']['c_train'])
plt.plot(x, readings['rbf']['c_valid'])
plt.plot(x, readings['rbf']['c_test'])
plt.xlabel("C Values")
plt.ylabel("Accuracy (%)")
plt.legend(['Training', 'Validation', 'Test'])
plt.title("Accuracy with RBF Kernel")
plt.show()

train_conf = confusion_matrix(train_label, predicted_label_train)
test_conf = confusion_matrix(test_label, predicted_label_test)
print(train_conf)
print(test_conf)

# For Creating Classifcation Report for MLR
print(classification_report(test_label, predicted_label_test, labels = list(range(0, 10))))
ans = []
for i in range(10):
    ans.append(np.mean(predicted_label_test[test_label == i] == test_label[test_label == i]))
ans
print(pd.DataFrame({'accuracy': ans}))

array_train = train_conf
df_cm_train = pd.DataFrame(array_train, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
plt.figure(figsize = (10,7))
sn.set(font_scale=1)
ax = plt.axes()
sn.heatmap(df_cm_train, annot=True,fmt="d")
ax.set_title('Confusion matrix for Train set')
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')

array_test = test_conf
df_cm_test = pd.DataFrame(array_test, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
plt.figure(figsize = (10,7))
sn.set(font_scale=1)
ax = plt.axes()
sn.heatmap(df_cm_test, annot=True,fmt="d")
ax.set_title('Confusion matrix for Test set')
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
