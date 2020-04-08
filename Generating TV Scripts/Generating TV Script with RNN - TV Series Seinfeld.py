#!/usr/bin/env python
# coding: utf-8

# # TV Script Generation : Seinfeld
# 
# Project Backgroud : Task to generate Seinfeld TV scripts using RNNs which the dataset source are from Kaggle (https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv), comprised of 9 seasons.  The Neural Network will generate a new TV script mimicking the original Seinfeld scripts, based on patterns it recognizes in the training.
# 

# ### 1. Load Dependencies Library 

# In[26]:


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter 
from string import punctuation
from torch.utils.data import TensorDataset, DataLoader


# ### 2. Load dataset

# In[78]:


data_dir = './data/Seinfeld_Scripts.txt'
input_file = os.path.join(data_dir)
with open(input_file, "r") as f:
    text = f.read()


# ### 3. Data Exploration
# Play around with `view_line_range` to view different parts of the data. This will give you a sense of the data you'll be working with. You can see, for example, that it is all lowercase text, and each new line of dialogue is separated by a newline character `\n`.

# In[49]:


view_line_range = (0, 10)


print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

lines = text.split('\n')
print('Number of lines: {}'.format(len(lines)))
word_count_line = [len(line.split()) for line in lines]
print('Average number of words in each line: {}'.format(np.average(word_count_line)))

print()
print('The lines {} to {}:'.format(*view_line_range))
print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))


# 
# ### 4. Data Pre-processing Functions
# First, we will build a Lookup table which consists of two dictionaries:
# - `vocab_to_int` (Word to id dictionary)
# - `int_to_vocab` (ID to id dictionary)
# 

# In[34]:


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # Add a counter of the words in the text provided
    word_count = Counter(text)
    # the word is sorted from most frequent to least frequent
    sorted_vocab = sorted(word_count, key=word_count.get, reverse=True)
    # creation of two dictionaries
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word : ii for ii, word in int_to_vocab.items()}

    return (vocab_to_int, int_to_vocab)


# ##### Tokenize Punctuation
# We'll be splitting the script into a word array using spaces as delimiters.  However, punctuations like periods and exclamation marks can create multiple ids for the same word. For example, "bye" and "bye!" would generate two different word ids.
# 
# Implement the function `token_lookup` to return a dict that will be used to tokenize symbols like "!" into "||Exclamation_Mark||".  Create a dictionary for the following symbols where the symbol is the key and value is the token:
# - Period ( **.** )
# - Comma ( **,** )
# - Quotation Mark ( **"** )
# - Semicolon ( **;** )
# - Exclamation mark ( **!** )
# - Question mark ( **?** )
# - Left Parentheses ( **(** )
# - Right Parentheses ( **)** )
# - Dash ( **-** )
# - Return ( **\n** )
# 
# This dictionary will be used to tokenize the symbols and add the delimiter (space) around it.  This separates each symbols as its own word, making it easier for the neural network to predict the next word. Make sure you don't use a value that could be confused as a word; for example, instead of using the value "dash", try using something like "||dash||".

# In[38]:


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    punctuation_dict = {
        '.':'||period||',
        ',':'||comma||',
        '"':'||quotation_mark||',
        ';':'||semicolon||',
        '!':'||exclamation_mark||',
        '?':'||question_mark||',
        '(':'||left_parentheses||',
        ')':'||right_parentheses||',
        '-':'||dash||',
        '\n':'||return||'
        }
    return punctuation_dict


# In[79]:


# Compile all the data preprocessing step

for key, token in token_lookup().items():
    text = text.replace(key,'{}'.format(token))

text = text.lower()
text = text.split()
vocab_to_int, int_to_vocab = create_lookup_tables(text + list({'PADDING': '<PAD>'}.values()))
int_text = [vocab_to_int[word] for word in text]


# ### 5. Load dataset into batches for processing
# 
# ##### Check Access to GPU

# In[7]:


# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')


# ##### Input
# [TensorDataset](http://pytorch.org/docs/master/data.html#torch.utils.data.TensorDataset) will be used in combination with [DataLoader](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader) to complie dataset, handle batching, shuffling, and other dataset iteration functions.
# 
# ##### Batching
# Implement the `batch_data` function to batch `words` data into chunks of size `batch_size` using the `TensorDataset` and `DataLoader` classes.
# 
# >Word batching by creating `feature_tensors` and `target_tensors` of the correct size and content for a given `sequence_length`.
# 
# Eg. We have these as input:
# ```
# words = [1, 2, 3, 4, 5, 6, 7]
# sequence_length = 4
# ```
# 
# First `feature_tensor` will contain these values:
# ```
# [1, 2, 3, 4]
# ```
# And the corresponding `target_tensor` will be the next "word"/tokenized word value:
# ```
# 5
# ```
# The Combination of `feature_tensor`, `target_tensor` will give the following:
# ```
# [2, 3, 4, 5]  # features
# 6             # target
# ```

# In[8]:


# Fuction to compile tensordataset & batching task
def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
#     n_batches = len(words)//batch_size
#     # only full batches
#     words = words[:n_batches*batch_size]
    y_len = len(words) - sequence_length
    x, y = [], []
    for idx in range(0, y_len):
        idx_end = sequence_length + idx
        x_batch = words[idx:idx_end]
        x.append(x_batch)
#         print("feature: ",x_batch)
        batch_y =  words[idx_end]
#         print("target: ", batch_y)    
        y.append(batch_y)    

    # create Tensor datasets
    data = TensorDataset(torch.from_numpy(np.asarray(x)), torch.from_numpy(np.asarray(y)))
    # make sure the SHUFFLE your training data
    data_loader = DataLoader(data, shuffle=False, batch_size=batch_size)
    # return a dataloader
    return data_loader    


# ##### Test functionality of the dataloader 
# 
# ###### Sizes
# Sample_x should be of size `(batch_size, sequence_length)` or (10, 5) in this case and sample_y should just have one dimension: batch_size (10). 
# 
# ###### Values
# 
# The targets, sample_y, are the *next* value in the ordered test_text data. So, for an input sequence `[ 28,  29,  30,  31,  32]` that ends with the value `32`, the corresponding output should be `33`.

# In[9]:


# test dataloader

test_text = range(50)
t_loader = batch_data(test_text, sequence_length=5, batch_size=10)

data_iter = iter(t_loader)
sample_x, sample_y = data_iter.next()

print(sample_x.shape)
print(sample_x)
print()
print(sample_y.shape)
print(sample_y)


# ### 6. Build RNN Neural Network

# In[80]:


class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5,lr=0.001):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        
        # define embedding layer        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        ## Define the LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        
        # set class variables
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # Define the final, fully-connected output layer
        self.fc = nn.Linear(hidden_dim, output_size)

        
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """

        batch_size = nn_input.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(nn_input)
        lstm_out, hidden = self.lstm(embeds, hidden)
    
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # dropout and fully-connected layer
        out = self.fc(lstm_out)
        
        # reshape into (batch_size, seq_length, output_size)
        out = out.view(batch_size, -1, self.output_size)
        # get last batch
        out = out[:, -1]

        return out, hidden

    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden


# ##### Define forward and backpropagation
# 
# **If a GPU is available, data should be moved to GPU device.**

# In[81]:


def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    # move model to GPU, if available
    if(train_on_gpu):
        rnn.cuda()
        
#     # Creating new variables for the hidden state, otherwise
#     # we'd backprop through the entire training history
    h = tuple([each.data for each in hidden])

    # zero accumulated gradients
    rnn.zero_grad()
    
    if(train_on_gpu):
        inputs, target = inp.cuda(), target.cuda()
#     print(h[0].data)
    
    # get predicted outputs
    output, h = rnn(inputs, h)
    
    # calculate loss
    loss = criterion(output, target)
    
#     optimizer.zero_grad()
    loss.backward()
    # 'clip_grad_norm' helps prevent the exploding gradient problem in RNNs / LSTMs
    nn.utils.clip_grad_norm_(rnn.parameters(), 5)

    optimizer.step()
    return loss.item(), h


# ### 7. Start Neural Network Training
# 
# 
# ##### Training Loop
# 

# In[17]:


def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    batch_losses = []
    
    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)
        
        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            
            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
            
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)          
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []

    # returns a trained rnn
    return rnn


# ##### Setting Hyperparameters for RNN
# 
# Set and train the neural network with the following parameters:
# - Set `sequence_length` to the length of a sequence.
# - Set `batch_size` to the batch size.
# - Set `num_epochs` to the number of epochs to train for.
# - Set `learning_rate` to the learning rate for an Adam optimizer.
# - Set `vocab_size` to the number of uniqe tokens in our vocabulary.
# - Set `output_size` to the desired size of the output.
# - Set `embedding_dim` to the embedding dimension; smaller than the vocab_size.
# - Set `hidden_dim` to the hidden dimension of your RNN.
# - Set `n_layers` to the number of layers/cells in your RNN.
# - Set `show_every_n_batches` to the number of batches at which the neural network should print progress.
# 

# In[13]:


# Data params

# Sequence Length
sequence_length = 10  # of words in a sequence
# Batch Size
batch_size = 128
train_loader = batch_data(int_text, sequence_length, batch_size)


# Training parameters

# Number of Epochs
num_epochs = 10
# Learning Rate
learning_rate = 0.001


# Model parameters

# Vocab size
vocab_size = len(vocab_to_int)
# Output size
output_size = vocab_size
# Embedding Dimension
embedding_dim = 250
# Hidden Dimension
hidden_dim = 300
# Number of RNN Layers
n_layers = 2

# Show stats for every n number of batches
show_every_n_batches = 500


# In[18]:


# Initiate training
# create model and move to gpu if available
rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
if train_on_gpu:
    rnn.cuda()

# defining loss and optimization functions for training
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# training the model
trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)


# ### 8. Generating TV Script
# 
# ### Generate Text
# To generate the text, the network needs to start with a single word and repeat its predictions until it reaches a set length. The `generate` function will takes a word id to start with, `prime_id`, and generates a set length of text, `predict_len`. It will use topk sampling to introduce some randomness in choosing the most likely next word, given an output set of word scores.

# In[20]:


def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, predict_len=100):
    """
    Generate text using the neural network
    :param decoder: The PyTorch Module that holds the trained neural network
    :param prime_id: The word id to start the first prediction
    :param int_to_vocab: Dict of word id keys to word values
    :param token_dict: Dict of puncuation tokens keys to puncuation values
    :param pad_value: The value used to pad a sequence
    :param predict_len: The length of text to generate
    :return: The generated text
    """
    rnn.eval()
    
    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, sequence_length), pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]
    
    for _ in range(predict_len):
        if train_on_gpu:
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)
        
        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0))
        
        # get the output of the rnn
        output, _ = rnn(current_seq, hidden)
        
        # get the next word probabilities
        p = F.softmax(output, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu
         
        # use top_k sampling to get the index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()
        
        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p/p.sum())
        
        # retrieve that word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)     
        
        # the generated word becomes the next "current sequence" and the cycle can continue
        current_seq = np.roll(current_seq, -1, 1)
        current_seq[-1][-1] = word_i
    
    gen_sentences = ' '.join(predicted)
    
    # Replace punctuation tokens
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
    gen_sentences = gen_sentences.replace('\n ', '\n')
    gen_sentences = gen_sentences.replace('( ', '(')
    
    # return all the sentences
    return gen_sentences


# ### Generate New Seinfeld Script
# Setting `gen_length` to the length of TV script desired to generate and set `prime_word` to one of the following to start the prediction:
# - "jerry"
# - "elaine"
# - "george"
# - "kramer"
# 
# One can also start with any other names found in the original text file.

# In[21]:


# run the cell multiple times to get different results!
gen_length = 500 # modify the length to your preference
prime_word = 'george' # name for starting the script

pad_word = helper.SPECIAL_WORDS['PADDING']
generated_script = generate(trained_rnn, vocab_to_int[prime_word + ':'], int_to_vocab, token_dict, vocab_to_int[pad_word], gen_length)
print(generated_script)


# ### 9. Save the output scripts

# In[22]:


# save script to a text file
f =  open("generated_seinfeld_script.txt","w")
f.write(generated_script)
f.close()


# ### The TV Script is Not Perfect
# TV script may not have the perfect sense as it is alternating lines of dialogue, but can generate better sense and accurate sequence with more training epochs and given more capacity to learn with hyperparameter tuning.
# 
# It can be seen that there are multiple characters that say (somewhat) complete sentences, as it takes quite a while to get good results, and often, using a smaller vocabulary (and discard uncommon words), or get more data will help. In this case, the Seinfeld dataset is about 3.4 MB, which is big enough for exploration.

# In[ ]:




