# Deep Neural Net
An easy to use library for constructing and training a neural network with an arbitrary number of layers and nodes. 
The following regularization techniques are included:
- Dropout
- Maxnorm
- Please see https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf for explanation of dropout and maxnorm 
## Installation
### Option 1: Clone repository
```
git clone https://github.com/StanleyFung/DeepNeuralNet.git
```
### Option 2:  Add it as a submodule in your own git project
```
git submodule add https://github.com/StanleyFung/DeepNeuralNet.git
```
Add the absolute path or relative path of this library from your project folder and import
```python
scriptpath = "../DeepNeuralNet/" # Relative
scriptpath = "~/github/DeepNeuralNet/" # Absolute
sys.path.append(os.path.abspath(scriptpath))
from deepneuralnet import *
```
### pip?
Will make into a pip package in the future 

# Usage
### Creating and training a model:
##### 1. Create a DNN. Use an integer for the identifier which will be used for saving and restoring the network
``` python
deepnn = DNN(1)
```
##### 2. Retrieve the data you want in a Pandas dataframe and create variable for name of label column
``` python
df = pd.DataFrame(...)
# Column name of correct labels Y
label = "label" 
```
##### 3. Split the data using 
``` python
# split_percent - Percentage of data to be used for training vs dev set. 
# 0.7 means 70 percent is used for training
(train_x, train_y, dev_x, dev_y) = DNN.split_data(df, label, split_percent =  0.7)
```
##### 4. Create a hyperparameters bundle
``` python
# Number of classes we wish to identify, >= 2
classes = 2 
# 4 layer NN with 10 nodes in first layer, 8 in second, etc.
layer_dims = [10, 8, 4, classes] 
learning_rate = 0.001
dropoout_keep_prob = 0.8
minibatch_size = 64
momentum = 0.97
dropout_maxnorm_clip = 3
hyperparams = DNN.create_hyperparameter_bundle(layer_dims=layer_dims, 
                                               learning_rate=learning_rate, 
                                               dropout_keep_prob = dropoout_keep_prob, 
                                               dropout_maxnorm_clip = dropout_maxnorm_clip, 
                                               beta1 = momentum, 
                                               minibatch_size = minibatch_size)
```
##### 5. Configure and construct the Neural Network 
```python
# train_x and train_y is result of DNN.split_data()
# Neural Net needs to know the number of inputs and outputs
deepnn.configure_graph(train_x, train_y)
```
##### 6. train model
```python
results = nn.train(train_x, train_y, dev_x, dev_y, 
                   num_epochs = 250, 
                   print_summary = True, 
                   checkpoint_interval = 200)
# checkpoint_interval will determine how many epochs to wait in-between save points
```
##### 7. predict
```python
predictions = deepnn.predict(test_data)
```
### NOTE: 
Use the following convenience methods to combine steps 3 - 6
```python 
set_hyperparams_split_data_configure_train(hyperparams, 
                                               df, 
                                               label, 
                                               num_epochs = 1000, 
                                               split_percent = 0.7, 
                                               print_summary = True, 
                                               checkpoint_interval = 100)
```

### Restoring a model and using it to further train or predict 
##### 1. Create a DNN. Identifier doesn't matter 
```python
restored = DNN()
```
##### 2. Restore model
```python
restored.restore_saved_model(identifier = 1, epoch = 200)
```
##### 3. Continue training or predict 
```python
results = restored.train(train_x, train_y, dev_x, dev_y, 
                      num_epochs = 250, print_summary = True, 
                      checkpoint_interval = 200)
prediction = restored.predict(test_data)
```
