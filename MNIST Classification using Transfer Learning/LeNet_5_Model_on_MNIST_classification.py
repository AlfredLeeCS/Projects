#!/usr/bin/env python
# coding: utf-8

# Project Description : A CV approach to solve MNIST classification problem using MXNet .The CV task is trained using LeNet-5 model.

# ### 1. Import Dependencies Library

# In[ ]:


from pathlib import Path
from mxnet import gluon, metric, autograd, init, nd
import os


# ### 2. Prepare train & test dataloader

# ##### To do transformation and normalization on the images data

# In[3]:


import os
from pathlib import Path
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms

def get_mnist_data(batch=128):

    mean, std = (0.13,),(0.31,)
    transform_fn = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean, std)])
    
    train_data = gluon.data.vision.datasets.MNIST(train=True)
    val_data = gluon.data.vision.datasets.MNIST(train=False)
    
    train_data = train_data.transform_first(transform_fn)
    val_data = val_data.transform_first(transform_fn)
    
    train_dataloader=gluon.data.DataLoader(train_data,
                                           batch_size = 128,
                                           shuffle=True)

    validation_dataloader=gluon.data.DataLoader(val_data,
                                                batch_size = 128,
                                                shuffle=False)
    
    return train_dataloader, validation_dataloader

t, v = get_mnist_data()


# Do forward iteration to get the respective images data and label
d, l = next(iter(t))


# ### 3. Model Training : Write the training loop

# In[ ]:


from time import time

# Define Loss function
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

def train(network, training_dataloader, batch_size, epochs):
    
    # Define training metrics
    train_acc = metric.Accuracy()
    
    #Define parameters needed for training : Optimizer & Learning Rate
    trainer = gluon.Trainer(network.collect_params(),'adam',{'learning_rate':0.002})

    # Write a training loop to feed forward, do back-propagation 
    # with the error identified to update the respective weights
    for epoch in range(epochs):
        train_loss = 0
        tic = time()
        for data, label in training_dataloader:
            with autograd.record():
                output = network(data)
                loss = loss_fn(output,label)
            loss.backward()
            trainer.step(batch_size)
            
            train_loss += loss.asnumpy().mean()
            train_acc.update(label,output)
        
        # Design to print epoch, loss, accuracy for every iteration
        print("Epoch(%d) Loss:%.3f Acc:%.3f "%(
            epoch, train_loss/len(training_dataloader),
            train_acc.get()[1]))

    return network, train_acc.get()[1]


# ##### Defining the model (neural network structure) & start the training process: 

# In[10]:


net = gluon.nn.Sequential()

# Add the hidden layers inside the Convolutional Neural Network with activation function
net.add(gluon.nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
        gluon.nn.MaxPool2D(pool_size=2, strides=2),
        gluon.nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
        gluon.nn.MaxPool2D(pool_size=2, strides=2),
        gluon.nn.Flatten(),
        gluon.nn.Dense(120, activation="relu"),
        gluon.nn.Dense(84, activation="relu"),
        # 10 output class
        gluon.nn.Dense(10))

# Model would need a initalizer
# Xavier is good and popular initializer
net.initialize(init=init.Xavier())


# Model training
# Training Step : 
# Batch Size : 128 ; Training Epochs : 5

n, ta = train(net, t, 128, 5)
d, l = next(iter(v))
p = (n(d).argmax(axis=1))


# ### 4. Model Validation 

# In[ ]:


def validate(network, validation_dataloader):
    """
    Should compute the accuracy of the network on the validation set.
    
    :param network: initialized gluon network to be trained
    :type network: gluon.Block
    
    :param validation_dataloader: the training DataLoader provides batches for data for every iteration
    :type validation_dataloader: gluon.data.DataLoader
    
    :return: validation accuracy
    :rtype: float
    """
    valid_acc = metric.Accuracy()
    for data, label in validation_dataloader:
        output = network(data)
        valid_acc.update(label, output)
    
    print("Validation Acc: %.3f "%(valid_acc.get()[1]))

#     raise NotImplementedError()
    
    return valid_acc.get()[1]


# ##### Complete with validation step to check on model performance

# In[12]:


validate(n, v) 


# In[ ]:


# Good to go with 98.9% validation accuracy

