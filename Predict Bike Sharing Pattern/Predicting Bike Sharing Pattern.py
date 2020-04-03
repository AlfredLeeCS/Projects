#!/usr/bin/env python
# coding: utf-8

# ### 1. Import Dependencies Libraries 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### 2. Load and prepare the data
# 

# In[2]:


data_path = 'Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)


# In[3]:


rides.head()


# In[11]:


rides[:10]


# ### Checking out the data
# 
# This dataset has the number of riders for each hour of each day from January 1 2011 to December 31 2012. The number of riders is split between casual and registered, summed up in the `cnt` column.
# 
# Below is a plot showing the number of bike riders over the first 10 days or so in the data set. (Some days don't have exactly 24 entries in the data set, so it's not exactly 10 days.) The hourly rentals here is pretty complicated. The weekends have lower over all ridership and there are spikes when people are biking to and from work during the week.

# In[4]:


rides[:24*10].plot(x='dteday', y='cnt')


# ### 3. Variables transformation

# #### Dummy variables using one-hot encoding
# Here are some categorical variables like season, weather, month. To include these in the model, will need to make binary dummy variables.

# In[5]:


dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
data.head()


# In[6]:


print(data.shape)


# #### Scaling target variables
# To make training network easier by standardizing every continuous variables, in which the variables will have zero mean and a standard deviation of 1.
# 

# In[7]:


quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary can be converted back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std


# In[8]:


scaled_features


# ### 4. Split data into training, testing, and validation sets
# 
# here, the last approximately 21 days to use as a test set after model training. This test set data will be used to make predictions and compare them with the actual number of riders.

# In[9]:


# Save data for approximately the last 21 days 
test_data = data[-21*24:]

# Now remove the test data from the data set 
data = data[:-21*24]

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]


# In[10]:


print(features.shape)
print(targets.shape)
print(test_features.shape)
print(test_targets.shape)


# We'll split the data into two sets, one for training and one for validating as the network is being trained. Since this is time series data, we'll train on historical data, then try to predict on future data (the validation set).

# In[11]:


# Hold out the last 60 days or so of the remaining data as a validation set
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]


# In[12]:


print(train_features.shape)
print(train_targets.shape)
print(val_features.shape)
print(val_targets.shape)


# ### 5.Building neural network
# 
# Elements in neural network: Forward pass, backwards pass and hyperparameters such as the learning rate, the number of hidden units, and the number of training passes need to be determined.
# 
# Ths network has two layers, a hidden layer and an output layer. The hidden layer will use the sigmoid function for activations. The output layer has only one node and is used for the regression, the output of the node is the same as the input of the node. The activation function is $f(x)=x$. 

# In[28]:


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))
                    
            
    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        ### Forward pass ###
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        ### Backward pass ###

        error = y - final_outputs
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        output_error_term = error*1 # f'(h) = 1 if activation = input
        weight_T = self.weights_hidden_to_output.transpose()
        hidden_error = np.dot(output_error_term,  weight_T)
        
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        
        delta_weights_i_h += hidden_error_term * X[:, None] 

        delta_weights_h_o += output_error_term * hidden_outputs[:, None]
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs
        
        return final_outputs



# Hyperparameters of the model need to be determined here

iterations = 3500
learning_rate = 0.9
hidden_nodes = 12
output_nodes = 1


# In[1]:


# Model loss function
def MSE(y, Y):
    return np.mean((y-Y)**2)


# ### 6. Model Training

# In[60]:


import sys

# Call the neural network model built
nn = NeuralNetwork()


N_i = train_features.shape[1]
network = NeuralNetwork(N_i, nn.hidden_nodes, nn.output_nodes, nn.learning_rate)

losses = {'train':[], 'validation':[]}
for ii in range(nn.iterations):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    X, y = train_features.ix[batch].values, train_targets.ix[batch]['cnt']
                             
    network.train(X, y)
    
    # Printing out the training progress
    train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(nn.iterations))                      + "% ... Training loss: " + str(train_loss)[:5]                      + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)


# In[61]:


plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
_ = plt.ylim()


# ### 7. Model Validation
# 
# Check on the test data.

# In[62]:


fig, ax = plt.subplots(figsize=(8,4))

mean, std = scaled_features['cnt']
predictions = network.run(test_features).T*std + mean
ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets['cnt']*std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)


# In[ ]:


# The model is acceptable with validation loss of 0.133

