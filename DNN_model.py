import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.metrics as sklearn
from tensorflow.python.framework import ops as tf_ops

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

def forward_propagation_no_dropout(X, parameters): 
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

def forward_propagation_with_dropout(X, parameters, keep_prob_tf): 
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> ... -> LINEAR -> SOFTMAX
    Arguments:
        X -- input dataset placeholder, of shape (input size, number of examples)
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"..."Wi", "bi"
                      the shapes are given in initialize_parameters
        keep_prob_tf - tensor for dropout probability place holder
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
            A = tf.nn.dropout(tf.nn.relu(Z), keep_prob_tf)            
        else:
            Z = tf.add(tf.matmul(W,A), b) 
            A = tf.nn.dropout(tf.nn.relu(Z), keep_prob_tf) 
 
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
    
    z_tf = forward_propagation_no_dropout(x_tf, params)
    
    with tf.Session() as sess:
        prediction = tf.argmax(z_tf)

        if is_binary_class:            
            prediction = tf.greater(tf.sigmoid(z_tf), 0.5)                    
                    
        result = sess.run(tf.cast(prediction, 'float'), feed_dict = {x_tf: X})
    
    # binary class returns array of shape (1, num_examples)
    # argmax in multi class reduces to 1D array for you
    if is_binary_class:
        result = result[0]

    return result

# Keys for dictionary returned by dnn.train
KEY_PARAMETERS = "params"
KEY_ACCURACY_TRAIN = "accuracy_train"
KEY_ACCURACY_TEST = "accuracy_test"
KEY_PRECISION = "precision"
KEY_RECALL = "recall"
KEY_F1 = "f1"

KEY_LAYER_DIMS = "layer_dims"
KEY_LEARNING_RATE = "learning_rate"
KEY_NUM_EPOCHS = "num_epochs"
KEY_KEEP_PROB = "keep_prob"
KEY_MINI_BATCH_SIZE = "minibatch_size"

def train_with_hyperparameter_bundle(x_train, y_train, x_test, y_test, bundle, print_summary = False):
    return train(x_train, y_train, x_test, y_test, bundle[KEY_LAYER_DIMS], bundle[KEY_LEARNING_RATE], bundle[KEY_NUM_EPOCHS], bundle[KEY_KEEP_PROB], bundle[KEY_MINI_BATCH_SIZE], print_summary)        

def train(X_train, Y_train, X_test, Y_test, layer_dims = [1], learning_rate = 0.0001, num_epochs = 5000, keep_prob = 1.0, minibatch_size = 64, print_summary = False):
    """
    Implements a L tensorflow neural network: LINEAR->RELU->LINEAR->RELU->...LINEAR->(SOFTMAX OR SIGMOID).
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
        keep_prob -- value from 0 - 1: probability a node is kept in the neural net during dropout
        learning_rate -- learning rate of the optimization
        num_epochs -- number of epochs of the optimization loop
        minibatch_size -- size of a minibatch
        print_summary -- True to print info and progress during and after training
    Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
    """
    assert(all(item >= 0 for item in layer_dims), "Number of nodes must be positive for all layers")
    
    is_binary_class = layer_dims[-1] <= 2    
    if print_summary:
        classification = "Binary" if layer_dims[-1] <= 2 else str(layer_dims[-1]) + "-class"
        print "Training " + classification + "neural network with hyperparameters:"
        print 'layer_dims: {0} keep_prob: {1} learning_rate: {2} num_epochs: {3} minibatch_size: {4}'.format(str(layer_dims), keep_prob, learning_rate, num_epochs, minibatch_size)
    
    tf_ops.reset_default_graph()
    keep_prob_tf = tf.placeholder(tf.float32)
    (n_x, m_train) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    X_place = tf.placeholder(dtype=tf.float32, shape = (n_x, None)) 
    Y_place = tf.placeholder(dtype=tf.float32, shape = (n_y, None)) 
    parameters = initialize_parameters(n_x, layer_dims)
    forward_prop_place = forward_propagation_with_dropout(X_place, parameters, keep_prob_tf)
    cost_func = compute_cost(forward_prop_place, Y_place, is_binary_class)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_func)
    result = {}

    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(math.floor(m_train / minibatch_size)) 
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
            for minibatch in minibatches: 
                (minibatch_X, minibatch_Y) = minibatch                
                _ , minibatch_cost = sess.run([optimizer, cost_func], feed_dict = { X_place: minibatch_X, Y_place: minibatch_Y, keep_prob_tf: keep_prob})        
                epoch_cost += minibatch_cost / num_minibatches
            
            if print_summary == True and epoch % 5 == 0: 
                costs.append(epoch_cost)

        parameters = sess.run(parameters)
                
        # Multi classification
        prediction = tf.argmax(forward_prop_place)
        predictions_correct = tf.equal(prediction, tf.argmax(Y_place))
        accuracy = tf.reduce_mean(tf.cast(predictions_correct, "float"))
        
        # Binary classification, Y_train = [[0 1 0 1 ...] contains labels of value 0 or 1 scalar
        if is_binary_class:
            prediction = tf.greater(tf.sigmoid(forward_prop_place),0.5)
            predictions_correct = tf.equal(prediction, tf.equal(Y_place,1.0))
            accuracy = tf.reduce_mean(tf.cast(predictions_correct, 'float') )        
  
        # Calculate accuracy on the test set
        train_accuracy = accuracy.eval({X_place: X_train, Y_place: Y_train, keep_prob_tf: 1.0})
        test_accuracy = accuracy.eval({X_place: X_test, Y_place: Y_test, keep_prob_tf: 1.0})      

        # Calculate precision, recall, and f1
        prediction_values_test = predict(X_test, parameters, is_binary_class)        

        precision = None
        recall = None
        f1score = None

        # http://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
        if is_binary_class:
            # make sure true labels are given as first parameter
            precision = sklearn.precision_score(Y_test[0], prediction_values_test)
            recall = sklearn.recall_score(Y_test[0], prediction_values_test)
            f1score = sklearn.f1_score(Y_test[0], prediction_values_test)
        else:
            precision = sklearn.precision_score(Y_test[0], prediction_values_test, average='micro')
            recall = sklearn.recall_score(Y_test[0], prediction_values_test, average='micro')
            f1score = sklearn.f1_score(Y_test[0], prediction_values_test, average='micro')

        if print_summary:
            print ("Done training!") 

            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

            print ("train_accuracy" + " : " + train_accuracy)
            print ("test_accuracy" + " : " + test_accuracy)  
            print ("precision" + " : " + precision)        
            print ("recall" + " : " + recall)        
            print ("f1score" + " : " + f1score)        

        result = {
            KEY_PARAMETERS: parameters,
            KEY_ACCURACY_TRAIN: train_accuracy,
            KEY_ACCURACY_TEST : test_accuracy,
            KEY_PRECISION : precision,
            KEY_RECALL : recall,
            KEY_F1: f1score
        }

    return result

def create_hyperparameter_bundle(layer_dims = [1], learning_rate = 0.0001, num_epochs = 5000, keep_prob = 1, minibatch_size = 64):
    bundle = {
        KEY_LAYER_DIMS: layer_dims,
        KEY_KEEP_PROB: keep_prob,
        KEY_LEARNING_RATE: learning_rate,
        KEY_NUM_EPOCHS: num_epochs,
        KEY_MINI_BATCH_SIZE: minibatch_size
    }
    return bundle

def kfold(df, label_column_name, bundle, k = 10.0, print_summary = False):
    layer_dims = bundle[KEY_LAYER_DIMS]
    is_binary_class = layer_dims[-1] <= 2
    m = len(df)
    folds = []
    permutation = list(np.random.permutation(m))
    shuffled = df.iloc[permutation]    
    fold_size = int(math.floor(m/k)) 

    for i in range(0, k):
        fold = None
        if i == k - 1:
            fold = shuffled[i*fold_size : m]        
        else:
            fold = shuffled[i*fold_size : (i+1) * fold_size]        
        folds.append(fold)  

    accuracy_test_sum = 0     

    for fold in folds:                
        test = fold 
        train = df.merge(fold, indicator=True, how='left')    
        train = train[train['_merge'] == 'left_only']
        train = train.drop('_merge', axis = 1)

        x_train = train.drop(label_column_name, axis = 1)
        x_test = train[label_column_name].T.values
        y_train = test.drop(label_column_name, axis = 1)
        y_test = test[label_column_name].values
        
        if !is_binary_class:
            y_test = dnn.one_hot_matrix(y_test, layer_dims[-1], axis = 0)

        model = train(x_train, y_train, x_test, y_test, , bundle[KEY_LEARNING_RATE], bundle[KEY_NUM_EPOCHS], bundle[KEY_KEEP_PROB], bundle[KEY_MINI_BATCH_SIZE], print_summary)        
        accuracy_test_sum += model[KEY_ACCURACY_TEST]

    return accuracy_test_sum/(1.0*len(folds))

