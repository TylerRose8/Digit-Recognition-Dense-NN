# Various tools for data manipulation. 
# Author: Bojian Xu, bojianxu@ewu.edu


import numpy as np
#import cupy as cp
import math

class MyUtils:
        
    def normalize_0_1(X):
        ''' Normalize the value of every feature into the [0,1] range, using formula: x = (x-x_min)/(x_max - x_min)
            1) First shift all feature values to be non-negative by subtracting the min of each column 
               if that min is negative.
            2) Then divide each feature value by the max of the column if that max is not zero. 
            
            X: n x d matrix of samples, excluding the x_0 = 1 feature. X can have negative numbers.
            return: the n x d matrix of samples where each feature value belongs to [0,1]
        '''
        n, d = X.shape
        X_norm = X.astype('float64') # Have a copy of the data in float

        for i in range(d):
            col_min = min(X_norm[:,i])
            col_max = max(X_norm[:,i])
            gap = col_max - col_min
            if gap:
                X_norm[:,i] = (X_norm[:,i] - col_min) / gap
            else:
                X_norm[:,i] = 0 #X_norm[:,i] - X_norm[:,i]
        
        return X_norm

    def normalize_neg1_pos1(X):
        ''' Normalize the value of every feature into the [-1,+1] range. 
            
            X: n x d matrix of samples, excluding the x_0 = 1 feature. X can have negative numbers.
            return: the n x d matrix of samples where each feature value belongs to [-1,1]
        '''
        # To be implemented
        n, d = X.shape
        X_norm = X.astype('float64') # Have a copy of the data in float

        for i in range(d):
            col_min = min(X_norm[:,i])
            col_max = max(X_norm[:,i])
            col_mid = (col_max + col_min) / 2
            gap = (col_max - col_min) / 2
            if gap:
                X_norm[:,i] = (X_norm[:,i] - col_mid) / gap
            else: 
                X_norm[:,i] = 0 #X_norm[:,i] - X_norm[:,i]

        return X_norm
        
    def z_transform(X, degree = 2):
        ''' Transforming traing samples to the Z space
            X: n x d matrix of samples, excluding the x_0 = 1 bias feature
            degree: the degree of the Z space
            return: the n x d' matrix of samples in the Z space, excluding the z_0 = 1 feature.
            It can be mathematically calculated: d' = \sum_{k=1}^{degree} (k+d-1) \choose (d-1)
        '''
        
        # Add your own code here. This function is not needed in NN implementation though. 

        if degree == 1:
            return X
        r_degree = degree
        _, d_xcols = X.shape
        B = []
        for i in range(0, r_degree):
            B.append(math.comb(i + d_xcols, d_xcols - 1))

        Z = np.copy(X)

        lexo = np.arange(np.sum(B))

        q_totalsizebucketsbeforeprev = 0
        p_prevbucketsize = d_xcols
        g_indexnewcolinZ = p_prevbucketsize
        cp_Z = cp.asarray(Z)
        cp_X = cp.asarray(X)

        # for each bucket needed
        for i in range(1, r_degree):
            # run next bucket
            for j in range(q_totalsizebucketsbeforeprev, p_prevbucketsize):
                for k in range(lexo[j], d_xcols):
                    temp = cp_Z[:, j] * cp_X[:, k]
                    cp_Z = cp.append(cp_Z, temp.reshape(-1, 1), axis=1)
                    lexo[g_indexnewcolinZ] = k
                    g_indexnewcolinZ += 1
            q_totalsizebucketsbeforeprev = p_prevbucketsize
            p_prevbucketsize = p_prevbucketsize + B[i]
        return cp_Z.get()
