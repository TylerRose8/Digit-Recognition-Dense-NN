# Place your EWU ID and name here

## delete the `pass` statement in every function below and add in your own code. 


import numpy as np
import cupy as cp

# Various math functions, including a collection of activation functions used in NN.

class MyMath:

    def tanh(x):
        ''' tanh function.
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh of the corresponding element in array x
        '''
        try:
            import cupy as cp
            return cp.vectorize(cp.tanh)(x)
        except ImportError:
            return np.vectorize(np.tanh)(x)


    def tanh_de(x):
        ''' Derivative of the tanh function.
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh derivative of the corresponding element in array x
        '''
        try:
            import cupy as cp
            return 1 - cp.vectorize(cp.tanh)(x)**2
        except ImportError:
            return 1 - np.vectorize(np.tanh)(x)**2


    def logis(x):
        ''' Logistic function.
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic of
                    the corresponding element in array x
        '''
        try:
            import cupy as cp
            return 1 / (1 + cp.exp(-1 * x))
        except ImportError:
            return 1 / (1 + cp.exp(-1 * x))


    def logis_de(x):
        ''' Derivative of the logistic unction.
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic derivative of
                    the corresponding element in array x
        '''
        try:
            import cupy as cp
            return cp.exp(-1 * x)/(1 + cp.exp(-1 * x))**2
        except ImportError:
            return np.exp(-1 * x)/(1 + np.exp(-1 * x))**2


    def iden(x):
        ''' Identity function
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the same as
                    the corresponding element in array x
        '''
        try:
            import cupy as cp
            cp.array(x)
        except ImportError:
            np.array(x)


    def iden_de(x):
        ''' The derivative of the identity function
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array of all ones of the same shape of x.
        '''
        try:
            import cupy as cp
            cp.ones(cp.array(x).shape)
        except ImportError:
            np.ones(np.array(x).shape)


    def relu(x):
        ''' The ReLU function
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the max of: zero vs. the corresponding element in x.
        '''
        try:
            import cupy as cp
            return cp.maximum(0, x)
        except ImportError:
            return np.maximum(0, x)


    def _relu_de_scaler(x):
        ''' The derivative of the ReLU function. Scaler version.

            x: a real number
            return: 1, if x > 0; 0, otherwise.
        '''
        return 1 if x > 0 else 0


    def relu_de(x):
        ''' The derivative of the ReLU function
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the _relu_de_scaler of the corresponding element in x.
        '''
        try:
            import cupy as cp
            return cp.maximum(0, x)
        except ImportError:
            return np.maximum(0, x)
