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
    KEY_MOMENTUM = "momentum"
    KEY_MAX_NORM_CLIP = "maxnorm_clip"

    OPS_COST = "cost"
    OPS_OPTIMIZER = "optimizer"

    # Utility methods for formating panda dataframes       
    @staticmethod
    def format_dataframe_for_training(df, label_column_name):
        x = None
        y = None

        if label_column_name:
            x = df.drop(label_column_name, axis = 1).values        
            y = df[label_column_name].values
            classification = len(set(y))
            y = DNN.__one_hot_matrix(y, classification)
        else:
            x = df.values

        return (x, y)

    def create_hyperparameter_bundle(layer_dims, learning_rate = 0.0001, num_epochs = 1000, keep_prob = 1, minibatch_size = 64, momentum = 0.97, maxnorm_clip = 4):
        bundle = {
            DNN.KEY_LAYER_DIMS: layer_dims,
            DNN.KEY_KEEP_PROB: keep_prob,
            DNN.KEY_LEARNING_RATE: learning_rate,
            DNN.KEY_NUM_EPOCHS: num_epochs,
            DNN.KEY_MINI_BATCH_SIZE: minibatch_size,            
            DNN.KEY_MOMENTUM: momentum,
            DNN.KEY_MAX_NORM_CLIP: maxnorm_clip
        }
        return bundle

    @staticmethod
    def split_data(df, label, split_percent = 0.7):                
        train = df.sample(frac=split_percent)
        dev = df.drop(train.index)
        (train_x, train_y) = DNN.format_dataframe_for_training(train, label)
        (dev_x, dev_y) = DNN.format_dataframe_for_training(dev, label)        
        print("train_x.shape: " + str(train_x.shape))
        print("train_y.shape: " + str(train_y.shape))
        print("dev_x.shape: " + str(dev_x.shape))
        print("dev_y.shape: " + str(dev_y.shape))
        return (train_x, train_y, dev_x, dev_y)

    def __init__(self):              
        self.__layerDims = None        
        self.__learningRate = None
        self.__numEpochs = None
        self.__dropoutKeepProb = None
        self.__minibatchSize = None
        self.__momentum = None
        self.__maxnormClip = None 
        self.__tf_keep_prob = None 
        self.__tf_X_place = None
        self.__tf_Y_place = None
        self.__tf_parameters = None
        self.__tf_Z_last = None
        self.__tf_cost_func = None            
        self.__tf_optimizer = None        
        self.__tf_prediction = None        
        self.__tf_true_values = None        
        self.__tf_predictions_correct = None
        self.__tf_accuracy= None
        self.__hyperparams_set = False
        self.__configured = False
    
    def set_hyperparameters(self, hyperparameters):
        """
        hyperparameters is a dictionary of {
                layerDims -- number of nodes in each layer, NOT including the input layers
                                    EG. layers_dims = [25, 7, 5, 1] 
                                    this would be a DNN of n_X -> 25 -> 7 -> 5 -> 1
                                    if number of nodes in last layer is > 1, we expect a multi class output                            
                learning_rate -- learning rate of the optimization
                num_epochs -- number of epochs of the optimization loop
                keep_prob -- value from 0 - 1: probability a node is kept in the neural net during dropout
                minibatch_size -- size of a minibatch    
            }
            generated from create_hyperparameter_bundle
        """    
        if len(hyperparameters[DNN.KEY_LAYER_DIMS]) == 0:
            print("layerDims can not be empty")
            return 

        if any(item < 0 for item in hyperparameters[DNN.KEY_LAYER_DIMS]):
            print("Number of nodes must be positive for all layers")
            return  

        if hyperparameters[DNN.KEY_LAYER_DIMS][-1] == 1:
            print("Number of nodes in last layer must be >= 2")
            return 

        print("Setting hyperparameters...")
        self.__layerDims = hyperparameters[DNN.KEY_LAYER_DIMS]        
        self.__learningRate = hyperparameters[DNN.KEY_LEARNING_RATE]
        self.__numEpochs = hyperparameters[DNN.KEY_NUM_EPOCHS]
        self.__dropoutKeepProb = hyperparameters[DNN.KEY_KEEP_PROB]
        self.__minibatchSize = hyperparameters[DNN.KEY_MINI_BATCH_SIZE] 
        self.__momentum = hyperparameters[DNN.KEY_MOMENTUM]  
        self.__maxnormClip = hyperparameters[DNN.KEY_MAX_NORM_CLIP]   
        self.__hyperparams_set = True        

    def __configure_graph(self, n_x, n_y, exp_id):        
        if not self.__hyperparams_set:
            print("Must set hyperparameters first using set_hyperparameters")
        else:
            print("Configuring graph...")
            tf_ops.reset_default_graph() 
            self.__tf_keep_prob = tf.placeholder(dtype=tf.float32)
            self.__tf_X_place = tf.placeholder(dtype=tf.float32, shape = (None, n_x)) 
            self.__tf_Y_place = tf.placeholder(dtype=tf.float32, shape = (None, n_y)) 
            self.__tf_parameters = self.__initialize_parameters(n_x)
            self.__tf_Z_last = self.__forward_propagation(self.__tf_X_place, self.__tf_parameters, self.__tf_keep_prob, True)        
            self.__tf_cost_func =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.__tf_Z_last, labels = self.__tf_Y_place))
            self.__tf_optimizer = tf.train.AdamOptimizer(learning_rate = self.__learningRate, beta1= self.__momentum).minimize(self.__tf_cost_func)                           
            self.__tf_prediction = tf.argmax(self.__tf_Z_last, axis = 1)
            self.__tf_true_values = tf.argmax(self.__tf_Y_place, axis = 1)
            self.__tf_predictions_correct = tf.equal(self.__tf_prediction, self.__tf_true_values)
            self.__tf_accuracy = tf.reduce_mean(tf.cast(self.__tf_predictions_correct, "float"))                
            self.__configured = True

    def set_hyperparams_split_data_configure_train(self, hyperparams, df, label, split_percent = 0.7, print_summary = True, exp_id = 1, checkpoint_interval = 200):        
        self.set_hyperparameters(hyperparams)
        return self.split_data_configure_train(df, label, split_percent, print_summary, exp_id, checkpoint_interval)

    def split_data_configure_train(self, df, label, split_percent = 0.7, print_summary = True, exp_id = 1, checkpoint_interval = 200):        
        """
        Splits data and trains model        
        Arguments:
            df -- pandas dataframe of shape (num_examples, num_inputs)
            label -- column name of labels
            split_percent -- amount to split for train and dev set
            print_summary -- True for printing progress while training 
        """
        print("Splitting data...")
        (train_x, train_y, dev_x, dev_y) = DNN.split_data(df, label, split_percent)        
        self.__configure_graph(train_x.shape[1], train_y.shape[1], exp_id)
        return self.train(train_x, train_y, dev_x, dev_y, print_summary, exp_id, checkpoint_interval)        

    def train(self, X_train, Y_train, X_test, Y_test, print_summary = True, exp_id = 1, checkpoint_interval = 200):
        """
        Implements a L tensorflow neural network: LINEAR->RELU->LINEAR->RELU->...LINEAR->(SOFTMAX OR SIGMOID).
        Arguments:
            X_train -- training set, of shape (number of training examples = m_train, input size = n_x)
            Y_train -- test set, of shape (m_train, n_y)
            X_test -- training set, of shape (m_test, n_x)
            Y_test -- test set, of shape (m_test, n_y)
            print_summary -- True to print info and progress during and after training
        Returns:
            result -- parameters learnt by the model. They can then be used to predict.
                      accuracy on training and test,
                      recall, precision, f1
        """
        if not self.__hyperparams_set or not self.__configured:
            print ("Must set hyper parameters and configure network for data first!")
            return None

        print("Experiment ID: " + str(exp_id))
        title = "Binary" if self.__layerDims[-1] <= 2 else str(self.__layerDims[-1]) + "-class"
        title += " classification neural network with hyperparameters:"
        print(title)
        print('layer_dims: {0} dropoutKeepProb: {1} learning_rate: {2} num_epochs: {3}'.format(str(self.__layerDims), 
            self.__dropoutKeepProb, 
            self.__learningRate,
            self.__numEpochs))

        print('minibatch_size: {0} momentum: {1} maxnormclip: {2}'.format(str(self.__minibatchSize), 
            self.__momentum, 
            self.__maxnormClip))
             
        (m_train_X, _) = X_train.shape
        (m_train_Y, _) = Y_train.shape

        if m_train_X != m_train_Y:
            print("Number of examples for training data in X and Y must be equal")
            return None

        m_train = m_train_X

        (m_test_X, _) = X_test.shape
        (m_test_Y, _) = Y_test.shape

        if m_test_X != m_test_Y:
            print("Number of examples for test data in X and Y must be equal")
            return None

        costs = []

        with tf.Session() as sess:            
            print("Initializing graph...")
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            # Initial saving of all variables and meta graph
            saver.save(sess, DNN.__getSavePath(exp_id), global_step = 0)
            print("Saving metagraph to " + DNN.__getSavePath(exp_id))

            print("Training underway...")
            for epoch in range(self.__numEpochs):
                epoch_cost = 0.

                if self.__minibatchSize < m_train:
                    num_minibatches = int(math.floor(m_train / self.__minibatchSize)) 
                    minibatches = self.__random_mini_batches(X_train, Y_train, self.__minibatchSize)
                    for minibatch in minibatches: 
                        (minibatch_X, minibatch_Y) = minibatch  

                        _ , minibatch_cost = sess.run([self.__tf_optimizer, self.__tf_cost_func], 
                            feed_dict = { self.__tf_X_place: minibatch_X, self.__tf_Y_place: minibatch_Y, 
                            self.__tf_keep_prob: self.__dropoutKeepProb})   

                        epoch_cost += minibatch_cost / num_minibatches
                else:
                    _ , batch_cost = sess.run([optimizer, cost_func], feed_dict = { self.__tf_X_place: X_train, 
                        self.__tf_Y_place: Y_train, self.__tf_keep_prob: self.__dropoutKeepProb})   

                    epoch_cost += batch_cost

                if print_summary == True and epoch % 20 == 0:
                    print("Cost after epoch %i: %f" % (epoch, epoch_cost)) 
                if print_summary == True and epoch % 5 == 0: 
                    costs.append(epoch_cost)

                if epoch > 0 and epoch % checkpoint_interval == 0:
                    saver.save(sess, DNN.__getSavePath(exp_id), global_step = epoch, write_meta_graph=False)
                    print("Saving checkpoint at epoch: " + str(epoch))
            
            saver.save(sess, DNN.__getSavePath(exp_id),  global_step = self.__numEpochs, write_meta_graph=False)
            print("Saving checkpoint at epoch: " + str(self.__numEpochs))
            print("Done Training!")                      
            
            parameters = sess.run(self.__tf_parameters)
                    
            # Calculate accuracy     
            train_accuracy = 100 * self.__tf_accuracy.eval({self.__tf_X_place: X_train, self.__tf_Y_place: Y_train, self.__tf_keep_prob: 1.0})        
            test_accuracy = 100 * self.__tf_accuracy.eval({self.__tf_X_place: X_test, self.__tf_Y_place: Y_test, self.__tf_keep_prob: 1.0})      
           
            prediction_values_test = self.__tf_prediction.eval({self.__tf_X_place: X_test, self.__tf_keep_prob: 1.0})                    
            true_values_test = self.__tf_true_values.eval({self.__tf_Y_place: Y_test, self.__tf_keep_prob: 1.0})
            
            precision = 100 * sklearn.precision_score(true_values_test, prediction_values_test, average='micro')
            recall = 100 * sklearn.recall_score(true_values_test, prediction_values_test, average='micro')
            f1score = 100 * sklearn.f1_score(true_values_test, prediction_values_test, average='micro')

            if print_summary:
                plt.plot(np.squeeze(costs))
                plt.ylabel('cost')
                plt.xlabel('iterations (per tens)')
                plt.title("Learning rate =" + str(self.__learningRate))
                plt.show()

                print("train_accuracy_percent" + " : " + str(train_accuracy))
                print("test_accuracy_percent" + " : " + str(test_accuracy))   

                print("precision_percent" + " : " + str(precision))        
                print("recall_percent" + " : " + str(recall))        
                print("f1score" + " : " + str(f1score))        

            result = {
                DNN.KEY_PARAMETERS: parameters,
                DNN.KEY_ACCURACY_TRAIN: train_accuracy,
                DNN.KEY_ACCURACY_TEST : test_accuracy,                
                DNN.KEY_PRECISION : precision,
                DNN.KEY_RECALL : recall,
                DNN.KEY_F1: f1score
            }
                    
            print('')                   

        return result

    def predict(self, X, params):
        """
        Outputs prediction for given test set and parameters from training 
        Arguments:
            X -- test data
            params -- parameters output from training 
        Returns:
            Predictions 
        """
        result = None      
        keep_prob_tf = tf.placeholder(tf.float32)     
        x_tf = tf.placeholder(dtype=tf.float32, shape = X.shape) 
        Z = self.__forward_propagation(X, params, keep_prob_tf, False)       
        
        with tf.Session() as sess:
            prediction = tf.argmax(Z, axis = 1)                                           
            result = sess.run(tf.cast(prediction, 'float'), feed_dict = {x_tf: X, keep_prob_tf: 1.0})

        return result

    def __initialize_parameters(self, n_x): 
        """
        Initializes parameters to build a neural network with tensorflow. 
        The shapes are:
        Input:
            n_x: number of inputs in first layer 
        Returns:
            W1 : [n_x, __layerDims[0]]
            b1 : [1, __layerDims[0]]
            W2 : [self.__layerDims[0], __layerDims[1]]
            b2 : [1, __layerDims[1]]
            .
            .
            .
            Wi : [self.__layerDims[0, __layerDims[i-1]]]
            bi : [1, __layerDims[i-1]]
        parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3... Wi, bi
        """
        parameters = {}
        for indx,item in enumerate(self.__layerDims):
            wKey = "W" + str(indx+1)
            bKey = "b" + str(indx+1)
            W = None
            b = None
            if indx == 0:            
                W = tf.get_variable(wKey, [n_x, item], initializer = tf.contrib.layers.xavier_initializer())
                b = tf.get_variable(bKey, [1, item], initializer = tf.zeros_initializer())            
            else:
                W = tf.get_variable(wKey, [self.__layerDims[indx - 1], item], initializer = tf.contrib.layers.xavier_initializer())
                b = tf.get_variable(bKey, [1, item], initializer = tf.zeros_initializer())            
            
            parameters[wKey] = W
            parameters[bKey] = b

        return parameters
        
    def __forward_propagation(self, X, parameters, keep_prob_tf, isTraining): 
        """
        Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> ... -> LINEAR -> SOFTMAX
        The following optimizations are included:
            Dropout 
            Batch Normalization 
            See https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
        Arguments:
            X -- input dataset placeholder, of shape (number of examples, input size)
            parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"..."Wi", "bi"
                          the shapes are given in initialize_parameters
            keep_prob_tf - tensor for dropout probability place holder - typical value ranges from 0.5 to 0.8
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
            
            if not isTraining:
                W = tf.convert_to_tensor(W)
                b = tf.convert_to_tensor(b)

            W = tf.clip_by_norm(W, self.__maxnormClip)

            if i == 0:
                Z = tf.add(tf.matmul(X, W), b)        
            else:     
                Z = tf.add(tf.matmul(A, W), b)                         
            
            A = tf.nn.dropout(tf.nn.relu(Z), keep_prob_tf) 
     
        return Z
    
    def __random_mini_batches(self, X, Y, minibatch_size): 
        """
        Creates a list of random minibatches from (X, Y)
        Arguments:
            X -- input data, of shape (number of examples, input size)
            Y -- true "label" vector (number of examples, n_y)            
        Returns:
            mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """
        (m, _) = X.shape        
        mini_batches = []

        # Step 1: Shuffle (X, Y)
        # To make your "random" minibatches
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation]
        shuffled_Y = Y[permutation]

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = int(math.floor(m/minibatch_size)) # number of mini batches of size self.__miniBatchSize in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[k*minibatch_size : (k+1) * minibatch_size]
            mini_batch_Y = shuffled_Y[k*minibatch_size : (k+1) * minibatch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < self.__miniBatchSize)
        if m % minibatch_size != 0:        
            mini_batch_X = shuffled_X[num_complete_minibatches * minibatch_size : num_complete_minibatches * minibatch_size + m % minibatch_size]
            mini_batch_Y = shuffled_Y[num_complete_minibatches * minibatch_size : num_complete_minibatches * minibatch_size + m % minibatch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        return mini_batches  

    def __one_hot_matrix(labels, C): 
        C = tf.constant(value = C, name = "C")    
        one_hot_matrix = tf.one_hot(labels, C, axis = -1)

        with tf.Session() as sess:
            one_hot = sess.run(one_hot_matrix)

        return one_hot  

    def __getSavePath(exp_id):
        PATH_SAVE = "./saved_model_" + str(exp_id) + "/dnn"
        return PATH_SAVE
