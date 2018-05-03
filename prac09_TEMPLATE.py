# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:51:35 2018

@author: Luke
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------

# Week 09 Exercise 1
def polynom(x,w):
    """
    performs a polynomial summation for values x with
    coefficients w and degree len(w) - 1

    parameters
    ----------
    x : np.ndarray or int
        values for x for which polynomial sums should be generated
    w : np.ndarray
        vector of polynomial coefficients with length n + 1

    returns
    -------
    an integer or an n-dimensional array with the same shape as x,
    containing polynomial sums for those values of x

    y = w[0] + w[1] * x^1 + w[2] * x^2 + ... + w[n] * x^n

        where
            n : length of w - 1
            y : same shape as x

    doctest
    -------
    >>> polynom(2, np.array([3, 0, 5]))
    23

    >>> polynom(np.array([2, 3]), np.array([3, 0, 5]))
    array([23, 48])

    >>> polynom(np.array([[1, 2], [2, 3], [4, 5]]), np.array([3, 0, 5]))
    array([[  8,  23],
           [ 23,  48],
           [ 83, 128]])
    """

    ###########
    # Steps:
    # 1. Set y to zero to start with
    # 2. Build up y as a polynomial sum
    #    for each coefficient in w
    # 3. Return y
    ############
    #
    # When you're done:
    #
    # Perform the necessary steps to ensure that y
    # is an array of zeros with the same shape as x
    # when x is a numpy array
    #
    ############

    
def test_polynom():
    x = 2
    w = np.array([3,0,5])
    print (polynom(x,w))

# ----------------------------------------

# Week 09 Exercise 2
#import matplotlib.pyplot as plt

def powerCol(x,n):
    '''
    returns
    -------
    y : ndarray
        if x is a scalar:
            y = [x^0  x^1  x^2  ...  x^n]

        if x is a vector [x0  x1  x2   ...  xm]:
            y = [[x0^0  x1^0  x2^0   ...  xm^0]
                 [x0^1  x1^1  x2^1   ...  xm^1]
                 [x0^2  x1^2  x2^2   ...  xm^2]
                 [ ...   ...   ...   ...   ...]
                 [x0^n  x1^n  x2^n   ...  xm^n]]

    parameters
    ----------
        x : np.ndarray or int
            values to bring to powers 0 to n
        n : int
            maximum power by which to multiply x

    doctest
    -------
    >>> power_col(2, 4)
    array([[  1.],
           [  2.],
           [  4.],
           [  8.],
           [ 16.]])

    >>> power_col(np.array([2, 1, -3]), 3)
    array([[  1.,   1.,   1.],
           [  2.,   1.,  -3.],
           [  4.,   1.,   9.],
           [  8.,   1., -27.]])
    '''

    ###################
    # Steps
    # 1. Create a numpy array y of zeros with the shape
    #       (n + 1, len(x))
    # 2. Populate the array (possibly row-by-row)
    ###################
    # When you're done:
    #
    # Perform any steps to ensure that x is
    # always a numpy array, even if a float/int is given
    #
    ###################


def test_powerCol():
    n = 5
    x = np.array([2,1,-3,4])
    print(powerCol(x,n))

# ----------------------------------------

# Week 09 Exercise 3
def challenge3():
    ##############################
    # GENERATE NOISY DATA POINTS

    
    ##############################
    # FIND BEST-FIT WEIGHTINGS W

    
    ######################################
    # PLOT ESTIMATES AGAINST ACTUAL VALUES
    

# ----------------------------------------

if __name__ == "__main__":
    challenge3()
    test_polynom()
    test_powerCol()

# ----------------------------------------
