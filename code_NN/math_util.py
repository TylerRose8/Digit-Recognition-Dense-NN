# Place your EWU ID and name here

## delete the `pass` statement in every function below and add in your own code. 


import numpy as np

# Various math functions, including a collection of activation functions used in NN.

class MyMath:

    def tanh(x):
        ''' tanh function.
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh of the corresponding element in array x
        '''
        return np.tanh(x)


    def tanh_de(x):
        ''' Derivative of the tanh function.
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh derivative of the corresponding element in array x
        '''
        return 1 - np.tanh(x)**2


    def logis(x):
        ''' Logistic function.
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic of
                    the corresponding element in array x
        '''

        return 1 / (1 + np.exp(-1 * x))


    def logis_de(x):
        ''' Derivative of the logistic unction.
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic derivative of
                    the corresponding element in array x
        '''
        return np.exp(-1 * x)/(1 + np.exp(-1 * x))**2


    def iden(x):
        ''' Identity function
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the same as
                    the corresponding element in array x
        '''

        return np.array(x)


    def iden_de(x):
        ''' The derivative of the identity function
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array of all ones of the same shape of x.
        '''
        return np.ones(np.array(x).shape)


    def relu(x):
        ''' The ReLU function
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the max of: zero vs. the corresponding element in x.
        '''
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
        return np.vectorize(MyMath._relu_de_scaler)(x)
