import numpy as np
import tensorflow as tf

def initialize_parameters(n_x, layer_dims): 
    """
    Initializes parameters to build a neural network with tensorflow. 
    The shapes are:
    Input:
        n_x: number of inputs in first layer 
        layer_dims: array containing number of nodes in each layer, not including input layer or output layer
    Returns:
        W1 : [layer_dims[0], n_x]
        b1 : [layer_dims[0], 1]
        W2 : [layer_dims[1], layer_dims[0]]
        b2 : [layer_dims[1], 1]
        .
        .
        .
        Wi : [layer_dims[i-1], layer_dims[0]]
        bi : [layer_dims[i-1], 1]
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3... Wi, bi
    """
    parameters = {}
    for indx,item in enumerate(layer_dims):
        wKey = "W" + str(indx+1)
        bKey = "b" + str(indx+1)
        W = None
        b = None
        if indx == 0:            
            W = tf.get_variable(wKey, [item,n_x], initializer = tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(bKey, [item,1], initializer = tf.zeros_initializer())            
        else:
            W = tf.get_variable(wKey, [item,layer_dims[indx - 1]], initializer = tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(bKey, [item,1], initializer = tf.zeros_initializer())            
        
        parameters[wKey] = W
        parameters[bKey] = b
                
    return parameters

def forward_propagation(X, parameters): 
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> ... -> LINEAR -> SOFTMAX
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"..."Wi", "bi"
                  the shapes are given in initialize_parameters
    Returns:
    Zi -- the output of the last LINEAR unit
    """
    Z = None
    A = None
    
    for i in range(0, len(parameters)/2):
        wKey = 'W' + str(i+1)
        bKey = 'b' + str(i+1)
        W = parameters[wKey]
        b = parameters[bKey]
        if i == 0:
            Z = tf.add(tf.matmul(W,X), b)
            A = tf.nn.relu(Z)
        else:
            Z = tf.add(tf.matmul(W,A), b) 
            A = tf.nn.relu(Z) 
    
    return Z

def compute_cost(Z_final, Y, isBinary): 
    """
    Computes the cost
    Arguments:
        Z_final -- output of forward propagation (output of the last LINEAR unit), of shape (n_y, number of examples)
        Y -- "true" labels vector placeholder, same shape as Z_final
    Returns:
        cost - Tensor of the cost function
    """
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits()
    # logits and labels must have the same shape
    # e.g. [batch_size, num_classes] and the same dtype (either float16, float32, or float64).
    # Z_final and Y is of shape (num_classes, batch_size)
    logits = tf.transpose(Z_final)
    labels = tf.transpose(Y)
    
    cost = None
    
    if isBinary: 
        # Use this for binary classification where Y is just an array of labels Eg. [0,1,0,1]
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
    else:
        # Use this for multi classification where Y consists of ONE HOT encodings
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    return cost