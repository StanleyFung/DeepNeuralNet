import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import DNN_operations as dnn_ops 
import DNN_utils as dnn_utils
from tensorflow.python.framework import ops as tf_ops

def train(X_train, Y_train, X_test, Y_test, layer_dims, learning_rate = 0.0001, num_epochs = 500, minibatch_size = 64, print_cost = True):
    """
    Implements a L tensorflow neural network: LINEAR->RELU->LINEAR->RELU->...LINEAR->SOFTMAX.
    Arguments:
        X_train -- training set, of shape (input size = n_x, number of training examples = m_train)
        Y_train -- test set, of shape (output size = n_y, number of training examples = m_train)

                # Binary classification, Y_train = [[0 1 0 1 ...] contains labels of value 0 or 1 scalar

                # For Multi Classification, one hot encoding
                # labels = np.array([1,2,3,0,2,1])
                # one_hot = one_hot_matrix(labels, C = 4, axis = 0)
                # Y_train = 
                #[[0. 0. 0. 1. 0. 0.] 
                # [1. 0. 0. 0. 0. 1.]
                # [0. 1. 0. 0. 1. 0.]
                # [0. 0. 1. 0. 0. 0.]]
                # Here 0 1 0 0, the first COLUMN represents 1

        X_test -- training set, of shape (input size = n_x, number of test examples = m_test)
        Y_test -- test set, of shape (output size = 6, number of test examples = m_test)
        layer_dims -- layer_dims is the number of nodes in each layer, NOT including the input layers
                    EG. layers_dims = [25, 7, 5, 1] 
                    this would be a DNN of n_X -> 25 -> 7 -> 5 -> 1
                    if number of nodes in last layer is > 1, we expect a multi class output
        learning_rate -- learning rate of the optimization
        num_epochs -- number of epochs of the optimization loop
        minibatch_size -- size of a minibatch
        print_cost -- True to print the cost every 100 epochs
    Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
    """
    tf_ops.reset_default_graph()
    is_binary_class = layer_dims[-1] <= 2
    (n_x, m_train) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    X_place = tf.placeholder(dtype=tf.float32, shape = (n_x, None)) 
    Y_place = tf.placeholder(dtype=tf.float32, shape = (n_y, None)) 
    parameters = dnn_ops.initialize_parameters(n_x, layer_dims)
    forward_prop_place = dnn_ops.forward_propagation(X_place, parameters)
    cost_func = dnn_ops.compute_cost(forward_prop_place, Y_place, is_binary_class)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_func)

    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(math.floor(m_train / minibatch_size)) 
            minibatches = dnn_utils.random_mini_batches(X_train, Y_train, minibatch_size)
            for minibatch in minibatches: 
                (minibatch_X, minibatch_Y) = minibatch                
                _ , minibatch_cost = sess.run([optimizer, cost_func], feed_dict = { X_place: minibatch_X, Y_place: minibatch_Y})        
                epoch_cost += minibatch_cost / num_minibatches
            
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0: 
                costs.append(epoch_cost)
    
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
                
        prediction = tf.argmax(forward_prop_place)
        predictions_correct = tf.equal(prediction, tf.argmax(Y_place))
        accuracy = tf.reduce_mean(tf.cast(predictions_correct, "float"))
        
        # Binary classification, Y_train = [[0 1 0 1 ...] contains labels of value 0 or 1 scalar
        if layer_dims[-1] == 1:
            prediction = tf.sigmoid(forward_prop_place)
            predicted_class = tf.greater(prediction,0.5)
            predictions_correct = tf.equal(predicted_class, tf.equal(Y_place,1.0))
            accuracy = tf.reduce_mean( tf.cast(predictions_correct, 'float') )        
              
        # Calculate accuracy on the test set
        print ("Train Accuracy:", accuracy.eval({X_place: X_train, Y_place: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X_place: X_test, Y_place: Y_test}))     
    
    return parameters

def predict(X, params_from_train, is_binary_class):
    result = None
    params = {}
    for i in range(0, len(params_from_train)/2):
        wKey = 'W' + str(i+1)
        bKey = 'b' + str(i+1)
        W = params_from_train[wKey]
        b = params_from_train[bKey]
        params[wKey] = tf.convert_to_tensor(W)
        params[bKey] = tf.convert_to_tensor(b)

    x_tf = tf.placeholder(dtype=tf.float32, shape = (X.shape[0], None)) 
    
    z_tf = dnn_ops.forward_propagation(x_tf, params)
    
    with tf.Session() as sess:
        prediction = tf.argmax(z_tf)

        if is_binary_class:            
            prediction = tf.greater(tf.sigmoid(z_tf), 0.5)                    
                    
        result = sess.run(prediction, feed_dict = {x_tf: X})

    return result