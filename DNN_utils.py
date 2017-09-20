import math
import numpy as np
import tensorflow as tf

def one_hot_matrix(labels, C, axis): 
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
    corresponds to the jth training example. So if example j had a label i. Then entry (i,j) will be 1.
    Arguments:
        labels -- vector containing the labels
        C -- number of classes, the depth of the one hot dimension
        axis -- features x depth if axis == -1 ELSE depth x features if axis == 0
    Returns:
        one_hot -- one hot matrix
    """
    C = tf.constant(value = C, name = "C")    
    one_hot_matrix = tf.one_hot(labels, C, axis = axis)

    with tf.Session() as sess:
        one_hot = sess.run(one_hot_matrix)
    
    return one_hot

def random_mini_batches(X, Y, mini_batch_size = 32): 
    """
    Creates a list of random minibatches from (X, Y)
    Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        mini_batch_size -- size of the mini-batches, integer
    Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[1]
    c = Y.shape[0]
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    # To make your "random" minibatches
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((c,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1) *mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:        
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : num_complete_minibatches * mini_batch_size + m % mini_batch_size]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : num_complete_minibatches * mini_batch_size + m % mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
