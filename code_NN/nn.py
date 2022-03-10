# Place your EWU ID and Name here. 

### Delete every `pass` statement below and add in your own code. 


# Implementation of the forwardfeed neural network using stachastic gradient descent via backpropagation
# Support parallel/batch mode: process every (mini)batch as a whole in one forward-feed/backtracking round trip. 


import numpy as np
import math
import math_util as mu
import nn_layer
from code_NN import math_util
import time


class NeuralNetwork:

    def __init__(self):
        self.layers = []  # the list of L+1 layers, including the input layer.
        self.L = -1  # Number of layers, excluding the input layer.
        # Initting it as -1 is to exclude the input layer in L.

    def add_layer(self, d=1, act='tanh'):
        ''' The newly added layer is always added AFTER all existing layers.
            The firstly added layer is the input layer.
            The most recently added layer is the output layer. 
            
            d: the number of nodes, excluding the bias node, which will always be added by the program. 
            act: the choice of activation function. The input layer will never use an activation function even if it is given. 
            
            So far, the set of supported activation functions are (new functions can be easily added into `math_util.py`): 
            - 'tanh': the tanh function
            - 'logis': the logistic function
            - 'iden': the identity function
            - 'relu': the ReLU function
        '''
        self.L += 1
        self.layers.append(nn_layer.NeuralLayer(d, act))

    def _init_weights(self):
        ''' Initialize every layer's edge weights with random numbers from [-1/sqrt(d),1/sqrt(d)], 
            where d is the number of nonbias node of the layer
        '''
        for i in range(1, self.L + 1):
            # self.layers[i].W = np.random.uniform(-1/math.sqrt(self.layers[i].d), 1/math.sqrt(self.layers[i].d), (self.layers[i-1].d+1, self.layers[i].d))
            weight_rng = np.random.default_rng(2142)
            self.layers[i].W = weight_rng.uniform(-1 / math.sqrt(self.layers[i].d), 1 / math.sqrt(self.layers[i].d),
                                                  (self.layers[i - 1].d + 1, self.layers[i].d))

    def _feed_forward(self, X, Y=None):
        cupy_available = False
        try:
            import cupy as np
            import numpy
            cupy_available = True
        except ImportError:
            cupy_available = False
            import numpy as np
        if cupy_available:
            X = np.array(X)
        time_start = time.time()
        self.layers[0].X = X
        for i in range(1, self.L + 1):
            if cupy_available:
                self.layers[i - 1].X = np.array(numpy.insert(self.layers[i - 1].X.get(), 0, 1, axis=1))
            else:
                self.layers[i - 1].X = np.insert(self.layers[i - 1].X, 0, 1, axis=1)
            current = self.layers[i]
            previous = self.layers[i - 1]
            current.W = np.array(current.W)
            current.S = previous.X @ current.W
            current.X = current.act(current.S)
        if cupy_available:
            self.layers[0].X = self.layers[0].X.get()
            for i in range(1, self.L + 1):
                self.layers[i].X = self.layers[i].X.get()
                self.layers[i].S = self.layers[i].S.get()
                self.layers[i].W = self.layers[i].W.get()
            import numpy as np
        print("Time for feed-forward: ", time.time() - time_start)
        if Y is not None:
            return np.sum(np.square(self.layers[self.L].X - Y)) / X.shape[0]

    def _back_propagation(self, y, N):
        try:
            import cupy as np
            cupy_available = True
        except ImportError:
            cupy_available = False
            import numpy as np
        time_start = time.time()
        if cupy_available:
            y = np.array(y)
            self.layers[self.L].X = np.array(self.layers[self.L].X)
            self.layers[self.L].W = np.array(self.layers[self.L].W)
            self.layers[self.L].S = np.array(self.layers[self.L].S)
        self.layers[self.L].Delta = 2 * (self.layers[self.L].X - y) * self.layers[self.L].act_de(self.layers[self.L].S)
        self.layers[self.L].G = np.einsum('ij,ik->jk', self.layers[self.L - 1].X, self.layers[self.L].Delta) * 1 / N
        for i in range(self.L - 1, 0, -1):
            if cupy_available:
                self.layers[i].X = np.array(self.layers[i].X)
                self.layers[i].W = np.array(self.layers[i].W)
                self.layers[i].S = np.array(self.layers[i].S)
            # self.layers[i-1].X = np.array(numpy.insert(self.layers[i-1].X, 0, 1, axis=1))
            layer = self.layers
            layer[i].Delta = layer[i].act_de(layer[i].S) * (layer[i + 1].Delta @ layer[i + 1].W[1:].T)
            layer[i].G = np.einsum('ij,ik->jk', layer[i - 1].X, layer[i].Delta) * 1 / N
        if cupy_available:
            self.layers[self.L].X = self.layers[self.L].X.get()
            self.layers[self.L].W = self.layers[self.L].W.get()
            self.layers[self.L].S = self.layers[self.L].S.get()
            self.layers[self.L].Delta = self.layers[self.L].Delta.get()
            self.layers[self.L].G = self.layers[self.L].G.get()
            for i in range(self.L - 1, 0, -1):
                self.layers[i].X = self.layers[i].X.get()
                self.layers[i].S = self.layers[i].S.get()
                self.layers[i].W = self.layers[i].W.get()
                self.layers[i].Delta = self.layers[i].Delta.get()
                self.layers[i].G = self.layers[i].G.get()
            import numpy as np
        print("Time for back-propagation: ", time.time() - time_start)

    def _update_weights(self, eta):
        time_start = time.time()
        for i in range(1, self.L + 1):
            self.layers[i].W = self.layers[i].W
            self.layers[i].G = self.layers[i].G
            left = self.layers[i].W
            right = eta * self.layers[i].G
            self.layers[i].W = np.array(left - right)
        print("Time for update weights: ", time.time() - time_start)

    def fit(self, X, Y, eta=0.01, iterations=1000, SGD=True, mini_batch_size=1):
        ''' Find the fitting weight matrices for every hidden layer and the output layer. Save them in the layers.

            X: n x d matrix of samples, where d >= 1 is the number of features in each training sample
            Y: n x k vector of lables, where k >= 1 is the number of classes in the multi-class classification
            eta: the learning rate used in gradient descent
            iterations: the maximum iterations used in gradient descent
            SGD: True - use SGD; False: use batch GD
            mini_batch_size: the size of each mini batch size, if SGD is True.
        '''
        self._init_weights()  # initialize the edge weights matrices with random numbers.
        errors = list()
        errors2 = list()
        if SGD:
            n, d = X.shape
            n_y, d_y = Y.shape
            np.reshape(X, (1, -1))
            np.reshape(Y, (1, -1))
            shuffleCount = 0
            time_start = time.time()
            for itr in range(iterations):
                batch = np.append(X, Y, axis=1)
                if shuffleCount == 0 or itr % n == 0:
                    # np.random.shuffle(batch)
                    shuffleCount += 1
                batch_X = batch[:, :-1]
                batch_X = np.split(batch_X, n / mini_batch_size)[itr % round(n / mini_batch_size - 0.25)]
                batch_y = batch[:, -1]
                batch_y = np.split(batch_y, n / mini_batch_size)[itr % round(n / mini_batch_size - 0.25)]

                for itr in range(batch_X.shape[0]):
                    error = self._feed_forward(batch_X[itr], Y)
                    errors.append(error)
                    # error2 = self.error(X, Y)
                    # errors2.append(error2)
                    print('error: ', error)
                    self._back_propagation(batch_y[itr], batch_X.shape[0])
                    self._update_weights(eta)
            print("Time for fit-batch: ", time.time() - time_start)
        else:
            time_start = time.time()
            for itr in range(iterations):
                print('iteration: ', itr)
                error = self._feed_forward(X, Y)
                errors.append(error)
                # error2 = self.error(X, Y)
                # errors2.append(error2)
                print('error: ', error)
                self._back_propagation(Y, X.shape[0])
                self._update_weights(eta)
            print("Time for fit-batch: ", time.time() - time_start)
        return errors, errors2
        # I will leave you to decide how you want to organize the rest of the code, but below is what I used and recommend. Decompose them into private components/functions.

        ## prep the data: add bias column; randomly shuffle data training set.

        ## for every iteration:
        #### get a minibatch and use it for:
        ######### forward feeding
        ######### calculate the error of this batch if you want to track/observe the error trend for viewing purpose.
        ######### back propagation to calculate the gradients of all the weights
        ######### use the gradients to update all the weight matrices.

    def predict(self, X):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column.
            
            return: n x 1 matrix, n is the number of samples, every row is the predicted class id.
         '''
        self._feed_forward(X)
        return np.argmax(self.layers[self.L].X, axis=1)

    def error(self, X, Y):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column. 
               n is the number of samples. 
               d is the number of (non-bias) features of each sample. 
            Y: n x k matrix, the labels of the input n samples. Each row is the label of one sample, 
               where only one entry is 1 and the rest are all 0. 
               Y[i,j]=1 indicates the ith sample belongs to class j.
               k is the number of classes. 
            
            return: the percentage of misclassfied samples
        '''

        Y = np.array(Y)
        error = np.count_nonzero(self.predict(X) - np.argmax(Y, axis=1))
        # error = np.sum(np.square(self.layers[self.L].X - Y))
        return error / Y.shape[0]
