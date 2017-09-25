import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.metrics as sklearn
from tensorflow.python.framework import ops as tf_ops

class DNN():

    # Keys for dictionary returned by train
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

    # Utility methods for formating panda dataframes 
    def predict(X, params):
        """
        Outputs prediction for given test set and parameters from training 
        Arguments:
            X -- test data
            params -- parameters output from training 
        Returns:
            Predictions 
        """

        result = None        
        Z = None
        A = None

        numLayers = int(len(params)/2)
        classification = None
        for i in range(0, numLayers):
            wKey = 'W' + str(i+1)
            bKey = 'b' + str(i+1)
            W = params[wKey]
            b = params[bKey]
            W_tf = tf.convert_to_tensor(W)
            b_tf = tf.convert_to_tensor(b)

            if i == numLayers - 1:
                classification = len(b)

            if i == 0:
                Z = tf.add(tf.matmul(W_tf,X), b_tf)
                A = tf.nn.relu(Z)
            else:
                Z = tf.add(tf.matmul(W,A), b_tf) 
                A = tf.nn.relu(Z)        

        x_tf = tf.placeholder(dtype=tf.float32, shape = (X.shape[0], None))             
        isBinary = classification <= 2
        with tf.Session() as sess:
            prediction = tf.argmax(Z)

            if isBinary:            
                prediction = tf.greater(tf.sigmoid(Z), 0.5)                    
                        
            result = sess.run(tf.cast(prediction, 'float'), feed_dict = {x_tf: X})

        # binary class returns array of shape (1, num_examples)
        # argmax in multi class reduces to 1D array for you
        if isBinary:
            result = result[0]

        return result

    def one_hot_matrix(labels, C): 
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
        one_hot_matrix = tf.one_hot(labels, C, axis = 0)

        with tf.Session() as sess:
            one_hot = sess.run(one_hot_matrix)

        return one_hot
        
    def create_hyperparameter_bundle(layer_dims, learning_rate = 0.0001, num_epochs = 5000, keep_prob = 1, minibatch_size = 64):
        bundle = {
            DNN.KEY_LAYER_DIMS: layer_dims,
            DNN.KEY_KEEP_PROB: keep_prob,
            DNN.KEY_LEARNING_RATE: learning_rate,
            DNN.KEY_NUM_EPOCHS: num_epochs,
            DNN.KEY_MINI_BATCH_SIZE: minibatch_size
        }
        return bundle

    @staticmethod
    def format_dataframe_for_training(df, label_column_name, classification):
        x = None
        y = None

        if classification == 2:
            classification = 1

        if label_column_name and len(label_column_name) > 0:
            x = df.drop(label_column_name, axis = 1).T.values        
            y = df[label_column_name].values
            y = DNN.one_hot_matrix(y, classification)
        else:
            x = df.T.values

        return (x, y)

    @staticmethod
    def split_data(df, label, classification, split_percent = 0.7):                
        train = df.sample(frac=split_percent)
        dev = df.drop(train.index)
        (train_x, train_y) = DNN.format_dataframe_for_training(train, label, classification)
        (dev_x, dev_y) = DNN.format_dataframe_for_training(dev, label, classification)
        print("train_x.shape: " + str(train_x.shape))
        print("train_y.shape: " + str(train_y.shape))
        print("dev_x.shape: " + str(dev_x.shape))
        print("dev_y.shape: " + str(dev_y.shape))
        return (train_x, train_y, dev_x, dev_y)

    def __init__(self, hyperparameters):  
        """
        hyperparameters is a dictionary of {
                layerDims -- number of nodes in each layer, NOT including the input layers
                                    EG. layers_dims = [25, 7, 5, 1] 
                                    this would be a DNN of n_X -> 25 -> 7 -> 5 -> 1
                                    if number of nodes in last layer is > 1, we expect a multi class output
                classification -- number of nodes in last layer, number of classes we want to identify
                isBinary -- whether or not this is a binary classification    
                learning_rate -- learning rate of the optimization
                num_epochs -- number of epochs of the optimization loop
                keep_prob -- value from 0 - 1: probability a node is kept in the neural net during dropout
                minibatch_size -- size of a minibatch    
            }
            generated from create_hyperparameter_bundle
        """       
        self.__layerDims = hyperparameters[DNN.KEY_LAYER_DIMS]
                
        if len(self.__layerDims) == 0:
            print("layerDims can not be empty")
            return None

        if any(item < 0 for item in self.__layerDims):
            print("Number of nodes must be positive for all layers")
            return None 

        self.__classification = self.__layerDims[-1]
        if self.__classification == 2:
            self.__classification == 1
        self.__isBinary = self.__classification <= 2
        self.__learningRate = hyperparameters[DNN.KEY_LEARNING_RATE]
        self.__numEpochs = hyperparameters[DNN.KEY_NUM_EPOCHS]
        self.__dropoutKeepProb = hyperparameters[DNN.KEY_KEEP_PROB]
        self.__minibatchSize = hyperparameters[DNN.KEY_MINI_BATCH_SIZE]           
    
    def split_data_and_train(self, df, label, split_percent = 0.7, print_summary = True):        
        """
        Splits data and trains model        
        Arguments:
            df -- pandas dataframe of shape (num_examples, num_inputs)
            label -- column name of labels
            classification -- number of classes to classify - Eg. Binary classification is 1 or 2, multi n class is n 
            split_percent -- amount to split for train and dev set
            print_summary -- True for printing progress while training 
        """
        (train_x, train_y, dev_x, dev_y) = DNN.split_data(df, label, self.__classification, split_percent)
        return self.train(train_x, train_y, dev_x, dev_y, print_summary)        

    def train(self, X_train, Y_train, X_test, Y_test, print_summary = True):
        """
        Implements a L tensorflow neural network: LINEAR->RELU->LINEAR->RELU->...LINEAR->(SOFTMAX OR SIGMOID).
        Arguments:
            X_train -- training set, of shape (input size = n_x, number of training examples = m_train)
            Y_train -- test set, of shape (output size = n_y or numOfClasses, number of training examples = m_train)

                    # Binary classification, Y_train = [[0 1 0 1 ...] contains labels of value 0 or 1 scalar

                    # For Multi Classification, one hot encoding
                    # labels = np.array([1,2,3,0,2,1])
                    # one_hot = one_hot_matrix(labels, C = 4)
                    # Y_train = 
                    #[[0. 0. 0. 1. 0. 0.] 
                    # [1. 0. 0. 0. 0. 1.]
                    # [0. 1. 0. 0. 1. 0.]
                    # [0. 0. 1. 0. 0. 0.]]
                    # Here 0 1 0 0, the first COLUMN represents 1

            X_test -- training set, of shape (input size = n_x, number of test examples = m_test)
            Y_test -- test set, of shape (output size = 6, number of test examples = m_test)           
            print_summary -- True to print info and progress during and after training
        Returns:
            result -- parameters learnt by the model. They can then be used to predict.
                      accuracy on training and test,
                      recall, precision, f1
        """
        
        classification = "Binary" if self.__classification <= 2 else str(self.__classification) + "-class"
        classification += " classification"
        print(classification + " neural network with hyperparameters:")
        print('layer_dims: {0} dropoutKeepProb: {1} learning_rate: {2} num_epochs: {3} minibatch_size: {4}'.format(str(self.__layerDims), 
            self.__dropoutKeepProb, 
            self.__learningRate,
            self.__numEpochs, 
            self.__minibatchSize))
        
        tf_ops.reset_default_graph()
        keep_prob_tf = tf.placeholder(tf.float32)
        (n_x, m_train) = X_train.shape
        n_y = Y_train.shape[0]
        costs = []
        X_place = tf.placeholder(dtype=tf.float32, shape = (n_x, None)) 
        Y_place = tf.placeholder(dtype=tf.float32, shape = (n_y, None)) 
        parameters = self.__initialize_parameters(n_x)
        forward_prop_place = self.__forward_propagation_with_dropout(X_place, parameters, keep_prob_tf)
        cost_func = self.__compute_cost(forward_prop_place, Y_place)
        optimizer = tf.train.AdamOptimizer(learning_rate = self.__learningRate).minimize(cost_func)
        result = {}

        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            
            print("Training underway...")
            for epoch in range(self.__numEpochs):
                epoch_cost = 0.

                if self.__minibatchSize < m_train:
                    num_minibatches = int(math.floor(m_train / self.__minibatchSize)) 
                    minibatches = self.__random_mini_batches(X_train, Y_train, self.__minibatchSize)
                    for minibatch in minibatches: 
                        (minibatch_X, minibatch_Y) = minibatch  
                        _ , minibatch_cost = sess.run([optimizer, cost_func], feed_dict = { X_place: minibatch_X, Y_place: minibatch_Y, keep_prob_tf: self.__dropoutKeepProb})        
                        epoch_cost += minibatch_cost / num_minibatches
                else:
                    _ , batch_cost = sess.run([optimizer, cost_func], feed_dict = { X_place: X_train, Y_place: Y_train, keep_prob_tf: self.__dropoutKeepProb})        
                    epoch_cost += batch_cost

                if print_summary == True and epoch % 20 == 0:
                    print("Cost after epoch %i: %f" % (epoch, epoch_cost)) 
                if print_summary == True and epoch % 5 == 0: 
                    costs.append(epoch_cost)

            parameters = sess.run(parameters)
                    
            # Multi classification
            prediction = tf.argmax(forward_prop_place)
            predictions_correct = tf.equal(prediction, tf.argmax(Y_place))
            accuracy = tf.reduce_mean(tf.cast(predictions_correct, "float"))
            
            # Binary classification, Y_train = [[0 1 0 1 ...] contains labels of value 0 or 1 scalar
            if self.__isBinary:
                prediction = tf.greater(tf.sigmoid(forward_prop_place),0.5)
                predictions_correct = tf.equal(prediction, tf.equal(Y_place,1.0))
                accuracy = tf.reduce_mean(tf.cast(predictions_correct, 'float') )        
      
            # Calculate accuracy 
            train_accuracy = accuracy.eval({X_place: X_train, Y_place: Y_train, keep_prob_tf: 1.0})        
            test_accuracy = accuracy.eval({X_place: X_test, Y_place: Y_test, keep_prob_tf: 1.0})      

            # Calculate precision, recall, and f1
            prediction_values_test = DNN.predict(X_test, parameters)        

            precision = None
            recall = None
            f1score = None

            # http://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
            if self.__isBinary:
                # make sure true labels are given as first parameter
                precision = sklearn.precision_score(Y_test[0], prediction_values_test)
                recall = sklearn.recall_score(Y_test[0], prediction_values_test)
                f1score = sklearn.f1_score(Y_test[0], prediction_values_test)
            else:
                precision = sklearn.precision_score(Y_test[0], prediction_values_test, average='micro')
                recall = sklearn.recall_score(Y_test[0], prediction_values_test, average='micro')
                f1score = sklearn.f1_score(Y_test[0], prediction_values_test, average='micro')

            if print_summary:
                plt.plot(np.squeeze(costs))
                plt.ylabel('cost')
                plt.xlabel('iterations (per tens)')
                plt.title("Learning rate =" + str(self.__learningRate))
                plt.show()

                print("train_accuracy" + " : " + str(train_accuracy))
                print("test_accuracy" + " : " + str(test_accuracy))  
                print("precision" + " : " + str(precision))        
                print("recall" + " : " + str(recall))        
                print("f1score" + " : " + str(f1score))        

            result = {
                DNN.KEY_PARAMETERS: parameters,
                DNN.KEY_ACCURACY_TRAIN: train_accuracy,
                DNN.KEY_ACCURACY_TEST : test_accuracy,
                DNN.KEY_PRECISION : precision,
                DNN.KEY_RECALL : recall,
                DNN.KEY_F1: f1score
            }
            
            print("Done training!")         
            print('')                   

        return result
    
    def __initialize_parameters(self, n_x): 
        """
        Initializes parameters to build a neural network with tensorflow. 
        The shapes are:
        Input:
            n_x: number of inputs in first layer 
        Returns:
            W1 : [__layerDims[0], n_x]
            b1 : [__layerDims[0], 1]
            W2 : [__layerDims[1], self.__layerDims[0]]
            b2 : [__layerDims[1], 1]
            .
            .
            .
            Wi : [__layerDims[i-1], self.__layerDims[0]]
            bi : [__layerDims[i-1], 1]
        parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3... Wi, bi
        """
        parameters = {}
        for indx,item in enumerate(self.__layerDims):
            wKey = "W" + str(indx+1)
            bKey = "b" + str(indx+1)
            W = None
            b = None
            if indx == 0:            
                W = tf.get_variable(wKey, [item,n_x], initializer = tf.contrib.layers.xavier_initializer())
                b = tf.get_variable(bKey, [item,1], initializer = tf.zeros_initializer())            
            else:
                W = tf.get_variable(wKey, [item,self.__layerDims[indx - 1]], initializer = tf.contrib.layers.xavier_initializer())
                b = tf.get_variable(bKey, [item,1], initializer = tf.zeros_initializer())            
            
            parameters[wKey] = W
            parameters[bKey] = b
                    
        return parameters
        
    def __forward_propagation_with_dropout(self, X, parameters, keep_prob_tf): 
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
        
        for i in range(0, int(len(parameters)/2)):
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
    
    def __compute_cost(self, Z_final, Y): 
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
        # e.g. [batch_size, num_classes] and the same dtype (either float16, float32, or float32).
        # Z_final and Y is of shape (num_classes, batch_size)
        logits = tf.transpose(Z_final)
        labels = tf.transpose(Y)    

        cost = None
        
        if self.__isBinary: 
            # Use this for binary classification where Y is just an array of labels Eg. [0,1,0,1]
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
        else:
            # Use this for multi classification where Y consists of ONE HOT encodings
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

        return cost
    
    def __random_mini_batches(self, X, Y, minibatch_size): 
        """
        Creates a list of random minibatches from (X, Y)
        Arguments:
            X -- input data, of shape (input size, number of examples)
            Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)            
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
        num_complete_minibatches = int(math.floor(m/minibatch_size)) # number of mini batches of size self.__miniBatchSize in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k*minibatch_size : (k+1) * minibatch_size]
            mini_batch_Y = shuffled_Y[:, k*minibatch_size : (k+1) * minibatch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < self.__miniBatchSize)
        if m % minibatch_size != 0:        
            mini_batch_X = shuffled_X[:, num_complete_minibatches * minibatch_size: num_complete_minibatches * minibatch_size + m % minibatch_size]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * minibatch_size : num_complete_minibatches * minibatch_size + m % minibatch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        return mini_batches