import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from datetime import datetime
import pickle

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return  1 / (1 + np.exp(-z))


def preprocess():
    """ Input:
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

     Some suggestions for preprocessing step:
     - feature selection"""

    digits = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples.
    # Your code here.
    train_data = np.array(digits['train0'])
    test_data = np.array(digits['test0'])
    train_label = [0 for _ in range(digits['train0'].shape[0])]
    test_label = [0 for _ in range(digits['test0'].shape[0])]

    for i in range(1, 10):
        train_data = np.vstack((train_data, digits['train' + str(i)]))
        train_label += [i for _ in range(digits['train' + str(i)].shape[0])]
        test_data = np.vstack((test_data, digits['test' + str(i)]))
        test_label += [i for _ in range(digits['test' + str(i)].shape[0])]

    train_data, validation_data, train_label, validation_label = train_test_split(train_data, train_label, test_size = (1 / 6))

    train_label = np.array(train_label).reshape(1, -1)
    validation_label = np.array(validation_label).reshape(1, -1)
    test_label = np.array(test_label).reshape(1, -1)

    train_data = train_data / 255
    validation_data = validation_data / 255
    test_data = test_data / 255

    # Feature selection
    # Your code here.
    selected_cols = []
    for col in range(train_data.shape[1]):
        if (train_data[:, col]).std() > 0:
            selected_cols.append(col)

    train_data = train_data[:, selected_cols]
    validation_data = validation_data[:, selected_cols]
    test_data = test_data[:, selected_cols]

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label, selected_cols


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    #
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    #
    # # Your code here
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array

    training_label = training_label.T
    aug = np.array([1 for _ in range(training_data.shape[0])]).reshape(-1, 1)
    input_data = np.hstack((training_data, aug))

    WTX1 = np.dot(input_data, w1.T)
    sigma1 = sigmoid(WTX1)

    input_hidden_data = np.append(sigma1,np.ones([sigma1.shape[0],1]),1)

    WTX2 = np.dot(input_hidden_data, w2.T)
    pred_val = sigmoid(WTX2)

    ## Response variable
    y = np.zeros((training_label.shape[0], pred_val.shape[1]))

    for i in range(training_label.shape[0]):
        y[i][np.int(training_label[i])] = 1

    err = -(np.multiply(y, (np.log(pred_val))) + np.multiply((1 - y), (np.log(1 - pred_val))))
    errFn = (np.sum(err)) / training_data.shape[0]


    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array

    obj_grad = np.array([])
    delta = pred_val - y

    init_grad_w2 = np.matmul(input_hidden_data.T,delta)

    f_1 = np.multiply(np.multiply((1 - input_hidden_data), input_hidden_data), np.matmul(delta, w2))
    init_grad_w1 = np.matmul(input_data.T, f_1)
    init_grad_w1 = init_grad_w1[:, 0 : n_hidden]

    w1_grad = (init_grad_w1.T + (lambdaval * w1)) / input_data.shape[0]
    w2_grad = (init_grad_w2.T + (lambdaval * w2)) / input_data.shape[0]

    obj_grad = np.concatenate((w1_grad.flatten(), w2_grad.flatten()), 0)
    obj_val = errFn + ((lambdaval / (2 * input_data.shape[0])) * (np.sum(np.square(w1)) + np.sum(np.square(w2))))

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    aug = np.array([1 for _ in range(data.shape[0])]).reshape(-1, 1)
    data = np.hstack((data, aug))
    out1 = np.matmul(w1, data.T)
    out1 = sigmoid(out1)

    aug = np.array([1 for _ in range(out1.shape[1])]).reshape(1, -1)
    out1 = np.vstack((out1, aug))
    out2 = np.matmul(w2, out1).T
    pred = sigmoid(out2)

    for i in range(pred.shape[0]):
        lab = np.argmax(pred[i, :])
        labels = np.append(labels, lab)

    return labels.reshape((1, -1))


"""**************Neural Network Script Starts here********************************"""
if __name__ == "__main__":
    np.random.seed(42)
    train_data, train_label, validation_data, validation_label, test_data, test_label, selected_features = preprocess()

    #  Train Neural Network

    # set the number of nodes in input unit (not including bias unit)
    n_input = train_data.shape[1]

    # set the number of nodes in hidden unit (not including bias unit)
    # n_hidden = 50

    # set the number of nodes in output unit
    n_class = 10

    hidden_units = [4, 8, 12, 16, 20]
    lambs = list(range(0, 61, 10))
    result = np.zeros((3, len(hidden_units), len(lambs)))
    times = np.zeros((len(hidden_units), len(lambs)))
    max_acc = 0
    best_w1 = None
    best_w2 = None
    best_hidden = 0
    best_lambda = -1
    best_lambda_index = -1

    for i in range(len(hidden_units)):
        n_hidden = hidden_units[i]
        print("Training", n_hidden, "units in the hidden layer...")
        for j in range(len(lambs)):
            lambdaval = lambs[j]
            print("    Trying lambda =", lambdaval, "for regularization...")
            # initialize the weights into some random matrices
            initial_w1 = initializeWeights(n_input, n_hidden)
            initial_w2 = initializeWeights(n_hidden, n_class)

            # unroll 2 weight matrices into single column vector
            initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

            # set the regularization hyper-parameter
            # lambdaval = 0

            args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

            # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

            opts = {'maxiter': 50}  # Preferred value.

            start = datetime.now()
            nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
            duration = datetime.now() - start

            times[i, j] = duration.seconds
            # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
            # and nnObjGradient. Check documentation for this function before you proceed.
            # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


            # Reshape nnParams from 1D vector into w1 and w2 matrices
            w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
            w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

            # Test the computed parameters

            predicted_label = nnPredict(w1, w2, train_data)

            # find the accuracy on Training Dataset
            # print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
            train_acc = 100 * np.mean((predicted_label == train_label).astype(float))
            result[0, i, j] = train_acc

            predicted_label = nnPredict(w1, w2, validation_data)

            # find the accuracy on Validation Dataset
            # print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
            valid_acc = 100 * np.mean((predicted_label == validation_label).astype(float))
            result[1, i, j] = valid_acc

            predicted_label = nnPredict(w1, w2, test_data)

            # find the accuracy on Test Dataset
            # print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
            test_acc = 100 * np.mean((predicted_label == test_label).astype(float))
            result[2, i, j] = test_acc

            if valid_acc > max_acc:
                max_acc = valid_acc
                best_w1 = w1
                best_w2 = w2
                best_hidden = n_hidden
                best_lambda = lambdaval
                best_lambda_index = j


    # predicted_label = nnPredict(w1, w2, test_data)
    print("Best Lambda:", best_lambda)
    print("Optimal no. of Hidden Units:", best_hidden)

    pdump = {
        "selected_features": selected_features,
        "n_hidden": best_hidden,
        "w1": best_w1,
        "w2": best_w2,
        "lambda": best_lambda
    }
    with open("params.pickle", "wb") as params_pickle:
        pickle.dump(pdump, params_pickle)

    predicted_label = nnPredict(best_w1, best_w2, test_data)
    # # find the accuracy on Test Dataset
    print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

    # Train Accuracy vs Lambda for various sizes of Hidden Layer
    plt.figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
    plt.style.use("default")
    plt.rc("font", size = 12)

    for row in range(len(hidden_units)):
        plt.plot(lambs, result[0, row, :], '--p')
        plt.errorbar(lambs, result[0, row, :], yerr = 1, fmt = 'none', ecolor = 'lightgray', elinewidth = 3, capsize = 0);

    plt.xlabel("$\lambda$")
    plt.ylabel("Accuracy (%)")
    plt.legend(hidden_units, frameon = False, title = "No. of Hidden Units")
    plt.title("Accuracy vs Lambda for various sizes of Hidden Layer - Training Data")
    plt.savefig("train_accuracy.png", dpi = 80, edgecolor = "k")
    # plt.show()

    # Validation Accuracy vs Lambda for various sizes of Hidden Layer
    plt.figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
    plt.style.use("default")
    plt.rc("font", size = 12)

    for row in range(len(hidden_units)):
        plt.plot(lambs, result[1, row, :], '--p')
        plt.errorbar(lambs, result[1, row, :], yerr = 1, fmt = 'none', ecolor = 'lightgray', elinewidth = 3, capsize = 0);

    plt.xlabel("$\lambda$")
    plt.ylabel("Accuracy (%)")
    plt.legend(hidden_units, frameon = False, title = "No. of Hidden Units")
    plt.title("Accuracy vs Lambda for various sizes of Hidden Layer - Validation Data")
    plt.savefig("validation_accuracy.png", dpi = 80, edgecolor = "k")

    # Test Accuracy vs Lambda for various sizes of Hidden Layer
    plt.figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
    plt.style.use("default")
    plt.rc("font", size = 12)

    for row in range(len(hidden_units)):
        plt.plot(lambs, result[2, row, :], '--p')
        plt.errorbar(lambs, result[2, row, :], yerr = 1, fmt = 'none', ecolor = 'lightgray', elinewidth = 3, capsize = 0);

    plt.xlabel("$\lambda$")
    plt.ylabel("Accuracy (%)")
    plt.legend(hidden_units, frameon = False, title = "No. of Hidden Units")
    plt.title("Accuracy vs Lambda for various sizes of Hidden Layer - Test Data")
    plt.savefig("test_accuracy.png", dpi = 80, edgecolor = "k")

    # Training Time vs No. of Units in the Hidden Layer
    plt.figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
    plt.rc("font", size = 12)

    plt.plot(hidden_units, times[:, best_lambda_index], '--p')

    plt.xlabel("No. of Units in the Hidden Layer")
    plt.ylabel("Time (secs)")
    plt.title("Training Time vs No. of Units in the Hidden Layer for $\lambda$ = " + str(best_lambda))
    plt.savefig("training_times.png", dpi = 80, edgecolor = "k")
