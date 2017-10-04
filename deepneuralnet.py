import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.metrics as sklearn
from tensorflow.python.framework import ops as tf_ops

class DNN():
    """
    Easy to use Deep Neural Network library with Dropout and Maxnorm using tensorflow.
    See https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf for explanation of dropout and maxnorm 
    
    Creating and training a model:
        1. Create a DNN with DNN(identifier = 1). Use an integer for the identifier 
        2. Retrieve the data you want in a Pandas dataframe 
        3. Split the data using DNN.split_data
        4. Create a hyperparameters bundle using DNN.create_hyperparameter_bundle
           Here you can specify the number of layers in the network and other parameters 
        5. Configure and construct the Neural Network using configure_graph
            train_x and train_y will be outputted from DNN.split_data
        6. train - model will be saved periodically at __get_save_path
        7. predict

        NOTE: Use set_hyperparams_split_data_configure_train or split_data_configure_train
            depending on your needs to combine steps 3-6

    Restoring a model and using it to further train or predict 
        1. Create a DNN using DNN()
        2. Call restore_saved_model
        3. At this point you can either 
            a)  split_data_train or split data yourself and call train
                If you want to change certain hyperparameters before training again,
                you can make a call to set_hyperparameters
            b)  predict
    """

    # Keys for dictionary returned by train
    KEY_PARAMETERS = "KEY_PARAMETERS"
    KEY_ACCURACY_TRAIN = "KEY_ACCURACY_TRAIN"
    KEY_ACCURACY_TEST = "KEY_ACCURACY_TEST"    
    KEY_PRECISION = "KEY_PRECISION"
    KEY_RECALL = "KEY_RECALL"
    KEY_F1 = "KEY_F1"  

    KEY_LAYER_DIMS = "KEY_LAYER_DIMS"
    KEY_LEARNING_RATE = "KEY_LEARNING_RATE"
    KEY_NUM_EPOCHS = "KEY_NUM_EPOCHS"
    KEY_DROPOUT_KEEP_PROB = "KEY_DROPOUT_KEEP_PROB"
    KEY_MINI_BATCH_SIZE = "KEY_MINI_BATCH_SIZE"
    KEY_ADAM_BETA1 = "KEY_ADAM_BETA1"
    KEY_MAX_NORM_CLIP = "KEY_MAX_NORM_CLIP"

    OPS_NUM_LAYERS = "OPS_NUM_LAYERS"
    OPS_X = "OPS_X"
    OPS_Y = "OPS_Y"
    OPS_LEARNING_RATE= "OPS_LEARNING_RATE"        
    OPS_DROPOUT_KEEP_PROB = "OPS_DROPOUT_KEEP_PROB"
    OPS_MINIBATCH_SIZE = "OPS_MINIBATCH_SIZE"
    OPS_ADAM_BETA1 = "OPS_ADAM_BETA1"
    OPS_MAXNORM_CLIP = "OPS_MAXNORM_CLIP"
    OPS_COST = "OPS_COST"
    OPS_OPTIMIZER = "OPS_OPTIMIZER"
    OPS_PREDICTION = "OPS_PREDICTION"
    OPS_TRUE_VALUES = "OPS_TRUE_VALUES"
    OPS_PREDICTIONS_CORRECT = "OPS_PREDICTIONS_CORRECT"
    OPS_ACCURACY = "OPS_ACCURACY"
    OPS_PREV_EPOCH = "OPS_PREV_EPOCH"

    def create_hyperparameter_bundle(layer_dims, learning_rate = 0.0001, dropout_keep_prob = 1.0, dropout_maxnorm_clip = 4, beta1 = 0.97, minibatch_size = 64):
        """
        Creates dictionary of hyperparameters       
        Arguments:
            See https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf for explanation of dropout and maxnorm 
            
            layer_dims - array of containing number of nodes in each layer.         
                        Eg. [5 4 classes] 
                        classes is the number of unique classes we want to label 
                        Eg. If label can be 0, 1, or 2, then classes = 3

                        Then there are 5 nodes in layer 1
                        4 nodes in layer 2
                        3 nodes in last layer(classes)
            learning_rate - learning rate alpha  
            dropout_keep_prob - Probability a node will be used during training 
            minibatch_size - Size of batch when training. 
                            Usual values are in form 2^n. Eg. 32, 64, 128, 256, 512,1024 etc
            beta1 - momentum used for accelerating learning rate in AdamOptimizer
            dropout_maxnorm_clip - Maximum value to clip weights by 
        """   
        bundle = {
            DNN.KEY_LAYER_DIMS: layer_dims,
            DNN.KEY_LEARNING_RATE: float(learning_rate),          
            DNN.KEY_DROPOUT_KEEP_PROB: float(dropout_keep_prob),
            DNN.KEY_MAX_NORM_CLIP: float(dropout_maxnorm_clip),          
            DNN.KEY_ADAM_BETA1: float(beta1),
            DNN.KEY_MINI_BATCH_SIZE: int(minibatch_size)
        }
        return bundle
    
    def split_data(df, label, split_percent = 0.7): 
        """
        split data into training and dev test sets         
        Arguments:
            df - pandas dataframe of shape (num_examples, num_inputs)
            label - name of the column you wish to classify 
            split_percent - Percentage of data to be used for training vs validation. 
                            EG. 0.7 means 70 percent is used for training
        """       
        train = df.sample(frac=split_percent)
        dev = df.drop(train.index)
        (train_x, train_y) = DNN.__format_dataframe_for_training(train, label)
        (dev_x, dev_y) = DNN.__format_dataframe_for_training(dev, label)        
        print("train_x.shape: " + str(train_x.shape))
        print("train_y.shape: " + str(train_y.shape))
        print("dev_x.shape: " + str(dev_x.shape))
        print("dev_y.shape: " + str(dev_y.shape))
        return (train_x, train_y, dev_x, dev_y)

    def __init__(self, identifier = 1):  
        """
        Class constructor         
        Arguments:
            identifier - Used for saving and restoring a previous model
        """
        tf_ops.reset_default_graph() 
        self.__identifier = identifier  
        self.__previous_epoch = 0
        self.__hyperparams_set = False
        self.__configured = False
    
    def set_hyperparameters(self, hyperparameters):
        """
        Restores previous saved model for further training or prediction
        Arguments:
            hyperparameters - dictionary generated from create_hyperparameter_bundle
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
        if DNN.KEY_LAYER_DIMS in hyperparameters:
            self.__layerDims = hyperparameters[DNN.KEY_LAYER_DIMS]             
            self.__tf_numLayers = tf.Variable(len(self.__layerDims),  trainable = False) 
        if DNN.KEY_LEARNING_RATE in hyperparameters:                    
            self.__tf_learningRate = tf.Variable(hyperparameters[DNN.KEY_LEARNING_RATE], trainable = False)
        if DNN.KEY_DROPOUT_KEEP_PROB in hyperparameters:
            self.__tf_dropoutKeepProb = tf.Variable(hyperparameters[DNN.KEY_DROPOUT_KEEP_PROB], trainable = False)             
        if DNN.KEY_MINI_BATCH_SIZE in hyperparameters:            
            self.__tf_minibatchSize = tf.Variable(hyperparameters[DNN.KEY_MINI_BATCH_SIZE], trainable = False)   
        if DNN.KEY_ADAM_BETA1 in hyperparameters:            
            self.__tf_adam_beta1 = tf.Variable(hyperparameters[DNN.KEY_ADAM_BETA1], trainable = False)        
        if DNN.KEY_MAX_NORM_CLIP in hyperparameters:
            self.__tf_maxnormClip = tf.Variable(hyperparameters[DNN.KEY_MAX_NORM_CLIP], trainable = False)               
        
        tf.add_to_collection(DNN.OPS_LEARNING_RATE, self.__tf_learningRate)            
        tf.add_to_collection(DNN.OPS_DROPOUT_KEEP_PROB, self.__tf_dropoutKeepProb)         
        tf.add_to_collection(DNN.OPS_MAXNORM_CLIP, self.__tf_maxnormClip)       
        tf.add_to_collection(DNN.OPS_ADAM_BETA1, self.__tf_adam_beta1)
        tf.add_to_collection(DNN.OPS_MINIBATCH_SIZE, self.__tf_minibatchSize)        
        self.__hyperparams_set = True        

    def configure_graph(self, train_x, train_y): 
        """
        Configures tensorflow placeholders and operations in order to construct the desired deep neural network
        Arguments:
            train_x - train_x returned from split_data
            train_y - train_y returned from split_data  
        """
        n_x = train_x.shape[1]
        n_y = train_y.shape[1]

        if not self.__hyperparams_set:
            print("Must set hyperparameters first using set_hyperparameters")
        else:
            print("Configuring graph...")          
            self.__tf_X_place = tf.placeholder(dtype=tf.float32, shape = (None, n_x)) 
            self.__tf_Y_place = tf.placeholder(dtype=tf.float32, shape = (None, n_y)) 
            self.__tf_parameters = self.__initialize_parameters(n_x)
            self.__tf_Z_last = self.__forward_propagation(True)        
            self.__tf_cost_func =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.__tf_Z_last, labels = self.__tf_Y_place))
            self.__tf_optimizer = tf.train.AdamOptimizer(learning_rate = self.__tf_learningRate, beta1=self.__tf_adam_beta1).minimize(self.__tf_cost_func)                           
            self.__tf_prediction = tf.argmax(self.__tf_Z_last, axis = 1)
            self.__tf_true_values = tf.argmax(self.__tf_Y_place, axis = 1)
            self.__tf_predictions_correct = tf.equal(self.__tf_prediction, self.__tf_true_values)
            self.__tf_accuracy = tf.reduce_mean(tf.cast(self.__tf_predictions_correct, "float"))                
            self.__tf_previous_epoch = tf.Variable(self.__previous_epoch, trainable = False)   
                            
            tf.add_to_collection(DNN.OPS_NUM_LAYERS, self.__tf_numLayers)
            tf.add_to_collection(DNN.OPS_X, self.__tf_X_place)
            tf.add_to_collection(DNN.OPS_Y, self.__tf_Y_place)
            tf.add_to_collection(DNN.OPS_COST, self.__tf_cost_func)
            tf.add_to_collection(DNN.OPS_OPTIMIZER, self.__tf_optimizer)
            tf.add_to_collection(DNN.OPS_PREDICTION, self.__tf_prediction)
            tf.add_to_collection(DNN.OPS_TRUE_VALUES, self.__tf_true_values)
            tf.add_to_collection(DNN.OPS_PREDICTIONS_CORRECT, self.__tf_predictions_correct)
            tf.add_to_collection(DNN.OPS_ACCURACY, self.__tf_accuracy)
            tf.add_to_collection(DNN.OPS_PREV_EPOCH, self.__tf_previous_epoch)
            self.__configured = True
    
    def restore_saved_model(self, identifier, epoch):   
        """
        Restores previous saved model for further training or prediction
        Arguments:
            identifier - id passed in constructor of DNN
            epoch - checkpoint you want to load
        """     
        with tf.Session() as sess:                                        
            checkpoint_dir = DNN.__get_save_path_with_epoch(identifier, epoch)           
            saver = tf.train.import_meta_graph(checkpoint_dir + ".meta")            
            saver.restore(sess, checkpoint_dir)         
            self.__identifier = identifier
            self.__tf_X_place = tf.get_collection(DNN.OPS_X)[0]
            self.__tf_Y_place = tf.get_collection(DNN.OPS_Y)[0]
            self.__tf_learningRate = tf.get_collection(DNN.OPS_LEARNING_RATE)[0]              
            self.__tf_dropoutKeepProb = tf.get_collection(DNN.OPS_DROPOUT_KEEP_PROB)[0]                 
            self.__tf_maxnormClip= tf.get_collection(DNN.OPS_MAXNORM_CLIP)[0]         
            self.__tf_adam_beta1 = tf.get_collection(DNN.OPS_ADAM_BETA1)[0]        
            self.__tf_minibatchSize= tf.get_collection(DNN.OPS_MINIBATCH_SIZE)[0]               
            self.__tf_cost_func = tf.get_collection(DNN.OPS_COST)[0]
            self.__tf_optimizer = tf.get_collection(DNN.OPS_OPTIMIZER)[0]
            self.__tf_prediction = tf.get_collection(DNN.OPS_PREDICTION)[0]
            self.__tf_true_values = tf.get_collection(DNN.OPS_TRUE_VALUES)[0]
            self.__tf_predictions_correct = tf.get_collection(DNN.OPS_PREDICTIONS_CORRECT)[0]
            self.__tf_accuracy = tf.get_collection(DNN.OPS_ACCURACY)[0]
            self.__tf_previous_epoch = tf.get_collection(DNN.OPS_PREV_EPOCH)[0]
            self.__previous_epoch = self.__tf_previous_epoch.eval()

            numLayers_tf = tf.get_collection(DNN.OPS_NUM_LAYERS)[0]
            numLayers = sess.run(numLayers_tf)
            parameters_tf = {}
            self.__layerDims = []
            for indx in range(0, numLayers):
                wKey = "W" + str(indx+1)
                bKey = "b" + str(indx+1)                
                parameters_tf[wKey] = tf.get_collection(wKey)[0]
                parameters_tf[bKey] = tf.get_collection(bKey)[0]
                self.__layerDims .append(parameters_tf[bKey].shape[1])

            self.__tf_parameters = parameters_tf 
            self.__parameters = sess.run(parameters_tf)           
            self.__configured = True
            self.__hyperparams_set = True                

    def set_hyperparams_split_data_configure_train(self, hyperparams, df, label, num_epochs, split_percent = 0.7, print_summary = True, checkpoint_interval = 200):        
        """
        Convenience method for setting hyperparams, configuring graph, splitting data, and training         
        Arguments:
            hyperparams - params from create_hyperparameter_bundle
            df - pandas dataframe of shape (num_examples, num_inputs)
            label - name of the column you wish to classify 
            num_epochs - number of iterations to train network
            split_percent - Percentage of data to be used for training vs validation. 
                            EG. 0.7 means 70 percent is used for training
            print_summary - True to print progress and summary of training when training 
            checkpoint_interval - Amount of epochs in between checkpoints when saving model 
                                EG. 200 means we will create a checkpoint every 200 epochs 
        """
        self.set_hyperparameters(hyperparams)
        return self.split_data_configure_train(df, label, num_epochs, split_percent, print_summary, checkpoint_interval)

    def split_data_configure_train(self, df, label, num_epochs, split_percent = 0.7, print_summary = True, checkpoint_interval = 200):        
        """
        Convenience method for configuring graph, splitting data, and training
        Arguments:
            hyperparams - params from create_hyperparameter_bundle
            df - pandas dataframe of shape (num_examples, num_inputs)
            label - name of the column you wish to classify 
            num_epochs - number of iterations to train network
            split_percent - Percentage of data to be used for training vs validation. 
                            EG. 0.7 means 70 percent is used for training
            print_summary - True to print progress and summary of training when training 
            checkpoint_interval - Amount of epochs in between checkpoints when saving model 
                                EG. 200 means we will create a checkpoint every 200 epochs 
        """
        print("Splitting data...")
        (train_x, train_y, dev_x, dev_y) = DNN.split_data(df, label, split_percent)        
        self.configure_graph(train_x, train_y)
        return self.train(train_x, train_y, dev_x, dev_y, num_epochs, print_summary, checkpoint_interval)        

    def split_data_train(self, df, label, num_epochs, split_percent = 0.7, print_summary = True, checkpoint_interval = 200):        
        """
        Splits data and trains model, use if restoring a saved model and want to keep training without
        having to set any hyperparameters again        
        Arguments:
            df -- pandas dataframe of shape (num_examples, num_inputs)
            label -- column name of labels
            num_epochs -- number of times to train
            split_percent -- amount to split for train and dev set
            print_summary -- True for printing progress while training 
            checkpoint_interval - Amount of epochs in between checkpoints when saving model 
                                EG. 200 means we will create a checkpoint every 200 epochs 
        """
        print("Splitting data...")
        (train_x, train_y, dev_x, dev_y) = DNN.split_data(df, label, split_percent)                
        return self.train(train_x, train_y, dev_x, dev_y, num_epochs, print_summary, checkpoint_interval)   

    def train(self, X_train, Y_train, X_test, Y_test, num_epochs, print_summary = True, checkpoint_interval = 200):
        """
        Implements a L tensorflow neural network: LINEAR->RELU->LINEAR->RELU->...LINEAR->(SOFTMAX OR SIGMOID).
        Arguments:
            X_train -- training set, of shape (number of training examples = m_train, input size = n_x)
            Y_train -- test set, of shape (m_train, n_y)
            X_test -- training set, of shape (m_test, n_x)
            Y_test -- test set, of shape (m_test, n_y)
            num_epochs -- number of times to train
            print_summary -- True for printing progress while training 
            checkpoint_interval - Amount of epochs in between checkpoints when saving model 
                                EG. 200 means we will create a checkpoint every 200 epochs 
        Returns:
            result -- parameters learnt by the model
                      accuracy on training and test,
                      recall, precision, f1
        """
        result = None
        with tf.Session() as sess: 
            if not self.__hyperparams_set:
                print ("Must set hyper parameters!")
                return None
            if not self.__configured:
                print ("Must configure network!")
                return None   
                          
            sess.run(tf.global_variables_initializer())

            print("Model ID: " + str(self.__identifier))
            title = "Binary" if self.__layerDims[-1] <= 2 else str(self.__layerDims[-1]) + "-class"
            title += " classification neural network with hyperparameters:"
            print(title)
            print('layer_dims: {0} learning_rate: {1}, dropoutKeepProb: {2}  num_epochs: {3}'.format(self.__layerDims,                 
                round(self.__tf_learningRate.eval(), 5),
                round(self.__tf_dropoutKeepProb.eval(), 5), 
                num_epochs))

            print('minibatch_size: {0} momentum: {1} maxnormclip: {2}'.format(self.__tf_minibatchSize.eval(), 
                round(self.__tf_adam_beta1.eval(), 5), 
                self.__tf_maxnormClip.eval()))
                 
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
             
            saver = tf.train.Saver()

            # Initial saving of all variables and meta graph
            saver.save(sess, DNN.__get_save_path(self.__identifier))
            print("Saving metagraph to " + DNN.__get_save_path(self.__identifier))
            
            feed_dict = {}
            savedEpoch = self.__previous_epoch

            if savedEpoch != 0:
                print("Resuming training from previous epoch of {0}".format(savedEpoch))            
            
            print("Training underway...")
            for epoch in range(num_epochs):
                epoch_cost = 0.     
                if self.__tf_minibatchSize.eval() < m_train:
                    num_minibatches = int(math.floor(m_train / self.__tf_minibatchSize.eval())) 
                    minibatches = self.__random_mini_batches(X_train, Y_train, self.__tf_minibatchSize.eval())
                    for minibatch in minibatches: 
                        (minibatch_X, minibatch_Y) = minibatch  
                        feed_dict[self.__tf_X_place] = minibatch_X
                        feed_dict[self.__tf_Y_place] = minibatch_Y
                        _ , minibatch_cost = sess.run([self.__tf_optimizer, self.__tf_cost_func], feed_dict = feed_dict)   
                        epoch_cost += minibatch_cost / num_minibatches
                else:
                    feed_dict[self.__tf_X_place] = X_train
                    feed_dict[self.__tf_Y_place] = Y_train
                    _ , batch_cost = sess.run([self.__tf_optimizer, self.__tf_cost_func], feed_dict = feed_dict)   
                    epoch_cost += batch_cost

                if print_summary == True and epoch % 20 == 0:
                    print("Cost after epoch %i: %f" % (epoch, epoch_cost)) 
                if print_summary == True and epoch % 5 == 0: 
                    costs.append(epoch_cost)

                if epoch > 0 and epoch % checkpoint_interval == 0: 
                    epochToSave = epoch + savedEpoch                                   
                    self.__tf_previous_epoch = tf.assign(self.__tf_previous_epoch, epochToSave)   
                    print("epoch to save {0}".format(self.__tf_previous_epoch.eval()))                  
                    saver.save(sess, DNN.__get_save_path(self.__identifier), global_step = epochToSave)                
                    print("Saving checkpoint at epoch: " + str(epochToSave))
            
            epochToSave = num_epochs + savedEpoch                   
            self.__tf_previous_epoch = tf.assign(self.__tf_previous_epoch, epochToSave) 
            print("epoch to save {0}".format(self.__tf_previous_epoch.eval()))
            saver.save(sess, DNN.__get_save_path(self.__identifier),  global_step = epochToSave)
            print("Saving checkpoint at epoch: " + str(epochToSave))
            print("Done Training!")                      
            
            self.__parameters = sess.run(self.__tf_parameters)
            feed_dict[self.__tf_X_place] = X_train
            feed_dict[self.__tf_Y_place] = Y_train
            feed_dict[self.__tf_dropoutKeepProb] = 1.0
            
            # Calculate accuracy     
            train_accuracy = 100 * self.__tf_accuracy.eval(feed_dict)   
            feed_dict[self.__tf_X_place] = X_test
            feed_dict[self.__tf_Y_place] = Y_test     
            test_accuracy = 100 * self.__tf_accuracy.eval(feed_dict)
           
            prediction_values_test = self.__tf_prediction.eval(feed_dict)
            true_values_test = self.__tf_true_values.eval(feed_dict)
            
            precision = 100 * sklearn.precision_score(true_values_test, prediction_values_test, average='micro')
            recall = 100 * sklearn.recall_score(true_values_test, prediction_values_test, average='micro')
            f1score = 100 * sklearn.f1_score(true_values_test, prediction_values_test, average='micro')

            if print_summary:
                plt.plot(np.squeeze(costs))
                plt.ylabel('cost')
                plt.xlabel('iterations (per tens)')
                plt.title("Learning rate =" + str(self.__tf_learningRate.eval()))
                plt.show()

                print("train_accuracy_percent" + " : " + str(train_accuracy))
                print("test_accuracy_percent" + " : " + str(test_accuracy))   

                print("precision_percent" + " : " + str(precision))        
                print("recall_percent" + " : " + str(recall))        
                print("f1score" + " : " + str(f1score))        

            result = {
                DNN.KEY_PARAMETERS: self.__parameters,
                DNN.KEY_ACCURACY_TRAIN: train_accuracy,
                DNN.KEY_ACCURACY_TEST : test_accuracy,                
                DNN.KEY_PRECISION : precision,
                DNN.KEY_RECALL : recall,
                DNN.KEY_F1: f1score
            }
                    
        print('')  
        return result                 

    def predict(self, X):
        """
        Outputs prediction for given test set  
        Arguments:
            X -- test data
        Returns:
            Predictions 
        """
        result = None       
        
        if self.__parameters is None:
            print("Cannot predict without training model first!")      
            return result

        Z = self.__forward_propagation(False)

        with tf.Session() as sess:            
            sess.run(tf.global_variables_initializer())

            feed_dict = {
                self.__tf_X_place: X,                 
                self.__tf_dropoutKeepProb: 1.0               
            }
            prediction = tf.argmax(Z, axis = 1)                                           
            result = sess.run(tf.cast(prediction, 'float'), feed_dict = feed_dict)

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
                     
            tf.add_to_collection(wKey, W)   
            tf.add_to_collection(bKey, b)  
            parameters[wKey] = W
            parameters[bKey] = b

        return parameters
        
    def __forward_propagation(self, isTraining): 
        """
        Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> ... -> LINEAR -> SOFTMAX
        The following optimizations are included:
            Dropout with Maxnorm             
            See https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
        Arguments:
           isTraining - whether or not we are training the model or use it for prediction 
                        if testing, we will take the parameters outputted from train()
                        else we will use __tf_parameters
        Returns:
            Zi -- the output of the last LINEAR unit
        """
        Z = None
        A = None  

        if isTraining:
            parameters = self.__tf_parameters
        else:
            parameters = self.__parameters

        for i in range(0, int(len(parameters)/2)):
            wKey = 'W' + str(i+1)
            bKey = 'b' + str(i+1)
            W = parameters[wKey]
            b = parameters[bKey]              
            
            if not isTraining:
                W = tf.convert_to_tensor(W)
                b = tf.convert_to_tensor(b)

            W = tf.clip_by_norm(W, self.__tf_maxnormClip)

            if i == 0:
                Z = tf.add(tf.matmul(self.__tf_X_place, W), b)        
            else:     
                Z = tf.add(tf.matmul(A, W), b)                         
            
            A = tf.nn.dropout(tf.nn.relu(Z), self.__tf_dropoutKeepProb) 
     
        return Z
    
    def __random_mini_batches(self, X, Y, minibatch_size): 
        """
        Creates a list of random minibatches from (X, Y)
        Arguments:
            X -- input data, of shape (number of examples, input size)
            Y -- true "label" vector (number of examples, n_y)  
            minibatch_size -- list of synchronous (mini_batch_X, mini_batch_Y)          
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

    def __one_hot_matrix(labels, classes): 
        """
        Creates one hot encoding for a column of labels 
        Arguments:
            labels - matrix of shape (num_examples, 1)
            classes - number of classes we wish to classify
                    EG. If label is either 0 or 1, then we have 2 possible classes
        """
        classes = tf.constant(value = classes, name = "classes")    
        one_hot_matrix = tf.one_hot(labels, classes, axis = -1)

        with tf.Session() as sess:
            one_hot = sess.run(one_hot_matrix)

        return one_hot  

    def __format_dataframe_for_training(df, label_column_name):
        """
        Extracts label column from panda dataframe and converts it into a one hot encoding 
        Arguments:
            df - pandas dataframe of shape (num_examples, num_inputs)
            label_column_name - name of column that contains the labels to be classified 
        """
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

    def __get_save_path(identifier):     
        """
        Returns save path for given identifier.
        Models are saved in format ./saved_model_{identifier}/dnn-{epoch}
        Arguments:
            identifier - id used in constructor of DNN
        """
        return "./saved_model_{0}/dnn".format(identifier)

    def __get_save_path_with_epoch(identifier, epoch):     
        """
        Returns save path for given identifier.
        Models are saved in format ./saved_model_{identifier}/dnn-{epoch}
        Arguments:
            identifier - id used in constructor of DNN
            epoch - epoch to load checkpoint from    
        """
        return "./saved_model_{0}/dnn-{1}".format(identifier, epoch)