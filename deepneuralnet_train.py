import math
import numpy as np
import pandas as pd
from deepneuralnet import *

KEY_KFOLD_AVERAGE_ACCURACY = "average_accuracy"

def kfold(df, label_column_name, bundle, k = 10.0, print_summary = True):    
    layer_dims = bundle[DNN.KEY_LAYER_DIMS]
    classification = layer_dims[-1]
    is_multi_class = layer_dims[-1] > 2
    m = len(df)
    folds = []
    permutation = list(np.random.permutation(m))
    shuffled = df.iloc[permutation]    
    fold_size = int(math.floor(m/k)) 

    for i in range(0, int(k)):
        fold = None
        if i == k - 1:
            fold = shuffled[i*fold_size : m]        
        else:
            fold = shuffled[i*fold_size : (i+1) * fold_size]        
        
        folds.append(fold)  

    accuracy_test_sum = 0     

    print("Starting K FOLD")

    i = 1
    for fold in folds: 
        print("Training fold " + str(i) + " / " + str(len(folds)))
        i += 1
        test = fold 
        train = df.merge(fold, indicator=True, how='left')    
        print("merge done")
        train = train[train['_merge'] == 'left_only']
        train = train.drop('_merge', axis = 1)        
        (x_train, y_train) = DNN.format_dataframe_for_training(train, label_column_name, classification)
        (x_test, y_test) = DNN.format_dataframe_for_training(test, label_column_name, classification)
        print("done formatting")
        deepNN = DeepNN(bundle)
        print("Created network")
        model = deepNN.train(x_train, y_train, x_test, y_test, print_summary)
        print("done training")
        accuracy_test_sum += model[DNN.KEY_ACCURACY_TEST]

    avg_accuracy = accuracy_test_sum/(1.0*len(folds))

    print("Average accuracy: " + str(avg_accuracy))
    print("Done K FOLD")
    print("")
    print("")

    return avg_accuracy