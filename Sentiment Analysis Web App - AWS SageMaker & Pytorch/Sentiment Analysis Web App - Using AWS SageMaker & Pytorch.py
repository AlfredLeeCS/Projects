#!/usr/bin/env python
# coding: utf-8

# ##### General Outline
# 
# 1. Download the data.
# 2. Process / Prepare the data.
# 3. Upload the processed data to S3.
# 4. Train a chosen model.
# 5. Test the trained model (typically using a batch transform job).
# 6. Deploy the trained model.
# 7. Use the deployed model.

# ### 1. Downloading the data
# 
# Data sources :  [IMDb dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
# 
# > Maas, Andrew L., et al. [Learning Word Vectors for Sentiment Analysis](http://ai.stanford.edu/~amaas/data/sentiment/). In _Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies_. Association for Computational Linguistics, 2011.

# In[1]:


get_ipython().run_line_magic('mkdir', '../data')
get_ipython().system('wget -O ../data/aclImdb_v1.tar.gz http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
get_ipython().system('tar -zxf ../data/aclImdb_v1.tar.gz -C ../data')


# ### 2. Import dependencies library

# In[ ]:


import os
import glob
import nltk
import re
import pickle
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.stem.porter import *
from bs4 import BeautifulSoup


# ### 3. Preparing and Processing the data
# 
# Reading in each of the reviews and combine them into a single input structure. Then, split the dataset into a training set and a testing set.

# In[2]:


def read_imdb_data(data_dir='../data/aclImdb'):
    data = {}
    labels = {}
    
    for data_type in ['train', 'test']:
        data[data_type] = {}
        labels[data_type] = {}
        
        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []
            
            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)
            
            for f in files:
                with open(f) as review:
                    data[data_type][sentiment].append(review.read())
                    # Here we represent a positive review by '1' and a negative review by '0'
                    labels[data_type][sentiment].append(1 if sentiment == 'pos' else 0)
                    
            assert len(data[data_type][sentiment]) == len(labels[data_type][sentiment]),                     "{}/{} data size does not match labels size".format(data_type, sentiment)
                
    return data, labels


# In[3]:


data, labels = read_imdb_data()
print("IMDB reviews: train = {} pos / {} neg, test = {} pos / {} neg".format(
            len(data['train']['pos']), len(data['train']['neg']),
            len(data['test']['pos']), len(data['test']['neg'])))


# Now that we've read the raw training and testing data from the downloaded dataset, we will combine the positive and negative reviews and shuffle the resulting records.

# In[4]:


def prepare_imdb_data(data, labels):
    """Prepare training and test sets from IMDb movie reviews."""
    
    #Combine positive and negative reviews and labels
    data_train = data['train']['pos'] + data['train']['neg']
    data_test = data['test']['pos'] + data['test']['neg']
    labels_train = labels['train']['pos'] + labels['train']['neg']
    labels_test = labels['test']['pos'] + labels['test']['neg']
    
    #Shuffle reviews and corresponding labels within training and test sets
    data_train, labels_train = shuffle(data_train, labels_train)
    data_test, labels_test = shuffle(data_test, labels_test)
    
    # Return a unified training data, test data, training labels, test labets
    return data_train, data_test, labels_train, labels_test


# In[5]:


train_X, test_X, train_y, test_y = prepare_imdb_data(data, labels)
print("IMDb reviews (combined): train = {}, test = {}".format(len(train_X), len(test_X)))


# In[6]:


# Check the preprocessed data

print(train_X[100])
print(train_y[100])


# To make sure that any html tags that appear are removed and in addition to word tokenization, words such as *entertained* and *entertaining* should considered the same with regard to sentiment analysis.

# In[7]:


def review_to_words(review):
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()
    
    text = BeautifulSoup(review, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case
    words = text.split() # Split string into words
    words = [w for w in words if w not in stopwords.words("english")] # Remove stopwords
    words = [PorterStemmer().stem(w) for w in words] # stem
    
    return words


# The `review_to_words` method defined above uses `BeautifulSoup` to remove any html tags that appear and uses the `nltk` package to tokenize the reviews.

# In[8]:


# Check out review (train_X[100])
review_to_words(train_X[100])


# In[10]:


# Preprocess data

cache_dir = os.path.join("../cache", "sentiment_analysis")  # where to store cache files
os.makedirs(cache_dir, exist_ok=True)  # ensure cache directory exists

def preprocess_data(data_train, data_test, labels_train, labels_test,
                    cache_dir=cache_dir, cache_file="preprocessed_data.pkl"):
    """Convert each review to words; read from cache if available."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay
    
    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Preprocess training and test data to obtain words for each review
        #words_train = list(map(review_to_words, data_train))
        #words_test = list(map(review_to_words, data_test))
        words_train = [review_to_words(review) for review in data_train]
        words_test = [review_to_words(review) for review in data_test]
        
        # Write to cache file for future runs
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test,
                              labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        words_train, words_test, labels_train, labels_test = (cache_data['words_train'],
                cache_data['words_test'], cache_data['labels_train'], cache_data['labels_test'])
    
    return words_train, words_test, labels_train, labels_test

train_X, test_X, train_y, test_y = preprocess_data(train_X, test_X, train_y, test_y)


# ### 4. Transform the data
# 
# To construct feature representation in which each word will be an integer. Some of the words that appear in the reviews occur very infrequently and so likely don't contain much information for the purposes of sentiment analysis. To deal withi this, the size of our working vocabulary will be fix and include only the words that appear most frequently. The infrequent words will be combined into a single category and label it as `1`.
# 
# Lastly, padding is needed for short reviews with the category 'no word' (which will be labelled `0`) and truncate long reviews.

# In[11]:


def build_dict(data, vocab_size = 5000):
    """Construct and return a dictionary mapping each of the most frequently appearing words to a unique integer."""
    
    word_count = {} # A dict storing the words that appear in the reviews along with how often they occur
    
    for sentence in data:
        for word in sentence:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
                
    # Sort the words found in `data` so that sorted_words[0] is the most frequently appearing word and
    # sorted_words[-1] is the least frequently appearing word.
    
    sorted_words = None
    
    sorted_words = sorted(word_count, key=word_count.get, reverse=True)
    
    word_dict = {} # This is what we are building, a dictionary that translates words into integers
    for idx, word in enumerate(sorted_words[:vocab_size - 2]): # The -2 is so that we save room for the 'no word'
        word_dict[word] = idx + 2                              # 'infrequent' labels
        
    return word_dict

word_dict = build_dict(train_X)


# In[13]:


# Check and see five most frequently appearing words in the training set.
{w for w, idx in word_dict.items() if idx < (5+2)}


# ##### Save `word_dict`
# 

# In[14]:


data_dir = '../data/pytorch' # The folder we will use for storing data
if not os.path.exists(data_dir): # Make sure that the folder exists
    os.makedirs(data_dir)


# In[15]:


with open(os.path.join(data_dir, 'word_dict.pkl'), "wb") as f:
    pickle.dump(word_dict, f)


# In[16]:


def convert_and_pad(word_dict, sentence, pad=500):
    NOWORD = 0 # We will use 0 to represent the 'no word' category
    INFREQ = 1 # and we use 1 to represent the infrequent words, i.e., words not appearing in word_dict
    
    working_sentence = [NOWORD] * pad
    
    for word_index, word in enumerate(sentence[:pad]):
        if word in word_dict:
            working_sentence[word_index] = word_dict[word]
        else:
            working_sentence[word_index] = INFREQ
            
    return working_sentence, min(len(sentence), pad)

def convert_and_pad_data(word_dict, data, pad=500):
    result = []
    lengths = []
    
    for sentence in data:
        converted, leng = convert_and_pad(word_dict, sentence, pad)
        result.append(converted)
        lengths.append(leng)
        
    return np.array(result), np.array(lengths)


# In[17]:


train_X, train_X_len = convert_and_pad_data(word_dict, train_X)
test_X, test_X_len = convert_and_pad_data(word_dict, test_X)


# In[18]:


# Check one of the processed reviews to make sure everything is working as intended.
train_X[1]


# ### 5. Upload the data to AWS S3
# 
# 
# ##### Save the processed training dataset locally
# 
# It is important to note the format of the data in which each row of the dataset has the form `label`, `length`, `review[500]` where `review[500]` is a sequence of `500` integers representing the words in the review.

# In[19]:


pd.concat([pd.DataFrame(train_y), pd.DataFrame(train_X_len), pd.DataFrame(train_X)], axis=1)         .to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)


# ##### Uploading the training data
# 
# 
# Next, we need to upload the training data to the SageMaker default S3 bucket so that we can provide access to it while training our model.

# In[20]:


import sagemaker

sagemaker_session = sagemaker.Session()

bucket = sagemaker_session.default_bucket()
prefix = 'sagemaker/sentiment_rnn'

role = sagemaker.get_execution_role()


# In[21]:


input_data = sagemaker_session.upload_data(path=data_dir, bucket=bucket, key_prefix=prefix)


# **NOTE:** The cell above uploads the entire contents of our data directory. This includes the `word_dict.pkl` file as we will need this later on when we create an endpoint that accepts an arbitrary review.

# ### 6. Build and Train the PyTorch Model
# 
# In particular, a model comprises three objects
# 
#  - Model Artifacts,
#  - Training Code, and
#  - Inference Code,
#  
# each of which interact with one another. We will be using containers provided by Amazon with the added benefit of being able to include our own custom code.
# 
# Implementing neural network in PyTorch along with a training script - model.py under train folder.

# In[22]:


get_ipython().system('pygmentize train/model.py')


# The important takeaway from the implementation is that there are three parameters that we may wish to tweak to improve the performance of our model. These are the embedding dimension, the hidden dimension and the size of the vocabulary. We will likely want to make these parameters configurable in the training script so that if we wish to modify them we do not need to modify the script itself.
# 
# First we will load a small portion of the training data set to use as a sample. It would be very time consuming to try and train the model completely in the notebook as we do not have access to a gpu and the compute instance that we are using is not particularly powerful, to see how our training script is behaving.

# In[23]:


import torch
import torch.utils.data

# Read in only the first 250 rows
train_sample = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None, names=None, nrows=250)

# Turn the input pandas dataframe into tensors
train_sample_y = torch.from_numpy(train_sample[[0]].values).float().squeeze()
train_sample_X = torch.from_numpy(train_sample.drop([0], axis=1).values).long()

# Build the dataset
train_sample_ds = torch.utils.data.TensorDataset(train_sample_X, train_sample_y)
# Build the dataloader
train_sample_dl = torch.utils.data.DataLoader(train_sample_ds, batch_size=50)


# In[24]:


def train(model, train_loader, epochs, optimizer, loss_fn, device):
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:         
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # TODO: Complete this train method to train the model provided.
            
            # to clear the accumulated gradients
            model.zero_grad()
            output = model.forward(batch_X)
            loss = loss_fn(output.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()
        print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))


# Supposing we have the training method above, we will test that it is working by writing a bit of code in the notebook that executes our training method on the small sample training set that we loaded earlier. The reason for doing this in the notebook is so that we have an opportunity to fix any errors that arise early when they are easier to diagnose.

# In[25]:


import torch.optim as optim
from train.model import LSTMClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(32, 100, 5000).to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = torch.nn.BCELoss()

train(model, train_sample_dl, 5, optimizer, loss_fn, device)


# In order to construct a PyTorch model using SageMaker we must provide SageMaker with a training script. We may optionally include a directory which will be copied to the container and from which our training code will be run. When the training container is executed it will check the uploaded directory (if there is one) for a `requirements.txt` file and install any required Python libraries, after which the training script will be run.

# In[26]:


from sagemaker.pytorch import PyTorch

estimator = PyTorch(entry_point="train.py",
                    source_dir="train",
                    role=role,
                    framework_version='0.4.0',
                    train_instance_count=1,
                    train_instance_type='ml.p2.xlarge',
                    hyperparameters={
                        'epochs': 10,
                        'hidden_dim': 200,
                    })


# In[27]:


estimator.fit({'training': input_data})


# ### 7. Testing the model
# 
# 
# #####  Deploy the model for testing
# 
# Now that we have trained our model, we would like to test it to see how it performs. Currently our model takes input of the form `review_length, review[500]` where `review[500]` is a sequence of `500` integers which describe the words present in the review, encoded using `word_dict`. SageMaker provides built-in inference code for models with simple inputs such as this.
# 
# There is one thing that we need to provide, To prepare function `model_fn()` that takes as its only parameter a path to the directory where the model artifacts are stored. This function must also be present in the python file which we specified as the entry point.
# 
# **NOTE**: When the built-in inference code is run it must import the `model_fn()` method from the `train.py` file with the training code is wrapped in a main guard ( ie, `if __name__ == '__main__':` )
# 
# **NOTE:** When deploying a model, SageMaker will launch an compute instance that will wait for data to be sent to it. As a result, this compute instance will continue to run until it is shutdown. To shut down the deployed endpoint when no longer using to avoid charges.
# 

# In[28]:


# Need to specify number of instances to be used & type of instance
predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')


# ### 8. Testing after deployment
# 
# Once deployed, we can read in the test data and send it off to our deployed model to get some results. Once we collect all of the results we can determine how accurate our model is.

# In[29]:


test_X = pd.concat([pd.DataFrame(test_X_len), pd.DataFrame(test_X)], axis=1)


# In[30]:


# We split the data into chunks and send each chunk seperately, accumulating the results.

def predict(data, rows=512):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = np.array([])
    for array in split_array:
        predictions = np.append(predictions, predictor.predict(array))
    
    return predictions


# In[31]:


predictions = predict(test_X.values)
predictions = [round(num) for num in predictions]


# In[32]:


from sklearn.metrics import accuracy_score
accuracy_score(test_y, predictions)


# ### 9. Another test cases
# 
# Ultimately we would like to be able to send our model an unprocessed review. That is, we would like to send the review itself as a string. For example, suppose we wish to send the following review to our model.

# In[33]:


test_review = 'The simplest pleasures in life are the best, and this film is one of them. Combining a rather basic storyline of love and adventure this movie transcends the usual weekend fair with wit and unmitigated charm.'


# In particular, we did two specific things to the provided reviews.
#  - Removed any html tags and stemmed the input
#  - Encoded the review as a sequence of integers using `word_dict`
#  
# In order process the review we will need to repeat these two steps.

# In[34]:


# Convert test_review into a form usable by the model and save the results in test_data
test_data = None
test_data_transform = review_to_words(test_review)
test_data = [np.array(convert_and_pad(word_dict, test_data_transform)[0])]


# Now that we have processed the review, we can send the resulting array to our model to predict the sentiment of the review.

# In[35]:


predictor.predict(test_data)


# Since the return value of our model is close to `1`, we can be certain that the review we submitted is positive.

# ##### Delete the endpoint
# 

# In[36]:


estimator.delete_endpoint()


# ### 10. Steps to deploy model for web app
# 
# Now that we know that our model is working, it's time to create some custom inference code so that we can send the model a review which has not been processed and have it determine the sentiment of the review.
# 
# As we saw above, by default the estimator which we created, when deployed, will use the entry script and directory which we provided when creating the model. However, since we now wish to accept a string as input and our model expects a processed review, we need to write some custom inference code.
# 
# We will store the code that we write in the `serve` directory. Provided in this directory is the `model.py` file that we used to construct our model, a `utils.py` file which contains the `review_to_words` and `convert_and_pad` pre-processing functions which we used during the initial data processing, and `predict.py`, the file which will contain our custom inference code. Note also that `requirements.txt` is present which will tell SageMaker what Python libraries are required by our custom inference code.
# 
# Four functions which the SageMaker inference container will use:
#  - `model_fn`: This function is the same function that we used in the training script and it tells SageMaker how to load our model.
#  - `input_fn`: This function receives the raw serialized input that has been sent to the model's endpoint and its job is to de-serialize and make the input available for the inference code.
#  - `output_fn`: This function takes the output of the inference code and its job is to serialize this output and return it to the caller of the model's endpoint.
#  - `predict_fn`: The heart of the inference script, this is where the actual prediction is done and is the function which you will need to complete.
# 
# For the simple website that we are constructing during this project, the `input_fn` and `output_fn` methods are relatively straightforward. We only require being able to accept a string as input and we expect to return a single value as output.

# In[37]:


get_ipython().system('pygmentize serve/predict.py')


# ##### Deploying the model
# 
# To construct a new PyTorchModel object which points to the model artifacts created during training and also points to the inference code that we wish to use and call the deploy method to launch the deployment container.
# 
# **NOTE**: The default behaviour for a deployed PyTorch model is to assume that any input passed to the predictor is a `numpy` array. In case to send a string, there will be a need to construct a simple wrapper around the `RealTimePredictor` class to accomodate simple strings.

# In[38]:


# check model_data location
model_data = estimator.model_data
model_data


# In[39]:


from sagemaker.predictor import RealTimePredictor
from sagemaker.pytorch import PyTorchModel

class StringPredictor(RealTimePredictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super(StringPredictor, self).__init__(endpoint_name, sagemaker_session, content_type='text/plain')

model = PyTorchModel(model_data=estimator.model_data,
                     role = role,
                     framework_version='0.4.0',
                     entry_point='predict.py',
                     source_dir='serve',
                     predictor_cls=StringPredictor)
predictor = model.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')


# ##### Testing the model
# 
# Now that we have deployed our model with the custom inference code, we should test to see if everything is working. Here we test our model by loading the first `250` positive and negative reviews and send them to the endpoint, then collect the results. The reason for only sending some of the data is that the amount of time it takes for our model to process the input and then perform inference is quite long and so testing the entire data set would be prohibitive.

# In[40]:


def test_reviews(data_dir='../data/aclImdb', stop=250):
    
    results = []
    ground = []
    
    # We make sure to test both positive and negative reviews    
    for sentiment in ['pos', 'neg']:
        
        path = os.path.join(data_dir, 'test', sentiment, '*.txt')
        files = glob.glob(path)
        
        files_read = 0
        
        print('Starting ', sentiment, ' files')
        
        # Iterate through the files and send them to the predictor
        for f in files:
            with open(f) as review:
                # First, we store the ground truth (was the review positive or negative)
                if sentiment == 'pos':
                    ground.append(1)
                else:
                    ground.append(0)
                # Read in the review and convert to 'utf-8' for transmission via HTTP
                review_input = review.read().encode('utf-8')
                # Send the review to the predictor and store the results
                results.append(float(predictor.predict(review_input)))
                
            # Sending reviews to our endpoint one at a time takes a while so we
            # only send a small number of reviews
            files_read += 1
            if files_read == stop:
                break
            
    return ground, results


# In[42]:


ground, results = test_reviews()


# In[43]:


from sklearn.metrics import accuracy_score
accuracy_score(ground, results)


# As an additional test, we can try sending the `test_review` that we looked at earlier.

# In[44]:


predictor.predict(test_review)


# Now that we know our endpoint is working as expected, we can set up the web page that will interact with it.

# ### 11. Use the model in the deployed web app
# 
# First we need to authenticate with AWS using an IAM role which included access to SageMaker endpoints. 
# 
# <img src="Web App Diagram.svg">
# 
# The diagram above gives an overview of how the various services will work together. On the far right is the model which we trained above and which is deployed using SageMaker. On the far left is our web app that collects a user's movie review, sends it off and expects a positive or negative sentiment in return.
# 
# We will construct a Lambda function, a straightforward Python function that can be executed whenever a specified event occurs. We will give this function permission to send and recieve data from a SageMaker endpoint.
# 
# Lastly, the method we will use to execute the Lambda function is a new endpoint that we will create using API Gateway. This endpoint will be a url that listens for data to be sent to it. Once it gets some data it will pass that data on to the Lambda function and then return whatever the Lambda function returns. Essentially it will act as an interface that lets our web app communicate with the Lambda function.
# 
# ##### Setting up a Lambda function
# 
# The first thing we are going to do is set up a Lambda function. This Lambda function will be executed whenever our public API has data sent to it. When it is executed it will receive the data, perform any sort of processing that is required, send the data (the review) to the SageMaker endpoint we've created and then return the result.
# 
# #####  Part A : Create an IAM Role for the Lambda function
# 
# Since we want the Lambda function to call a SageMaker endpoint, we need to make sure that it has permission to do so. To do this, we will construct a role that we can later give the Lambda function.
# 
# Using the AWS Console, navigate to the **IAM** page and click on **Roles**. Then, click on **Create role**. Make sure that the **AWS service** is the type of trusted entity selected and choose **Lambda** as the service that will use this role, then click **Next: Permissions**.
# 
# In the search box type `sagemaker` and select the check box next to the **AmazonSageMakerFullAccess** policy. Then, click on **Next: Review**.
# 
# Lastly, give this role a name. Make sure you use a name that you will remember later on, for example `LambdaSageMakerRole`. Then, click on **Create role**.
# 
# ##### Part B: Create a Lambda function
# 
# 
# Using the AWS Console, navigate to the AWS Lambda page and click on **Create a function**and select **Author from scratch** on the next page. Name the Lambda function, for example `sentiment_analysis_func` and make sure that the **Python 3.6** runtime is selected and then choose the role that created in the previous part. Then, click on **Create Function**.
# 
# 
# ```python
# # We need to use the low-level library to interact with SageMaker since the SageMaker API
# # is not available natively through Lambda.
# import boto3
# 
# def lambda_handler(event, context):
# 
#     # The SageMaker runtime is what allows us to invoke the endpoint that we've created.
#     runtime = boto3.Session().client('sagemaker-runtime')
# 
#     # Now we use the SageMaker runtime to invoke our endpoint, sending the review we were given
#     response = runtime.invoke_endpoint(EndpointName = 'sagemaker-pytorch-2020-04-14-13-52-49-456', # Endpoint name created
#                                        ContentType = 'text/plain',                 # The data format that is expected
#                                        Body = event['body'])                       # The actual review
# 
#     # The response is an HTTP response whose body contains the result of our inference
#     result = response['Body'].read().decode('utf-8')
# 
#     return {
#         'statusCode' : 200,
#         'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
#         'body' : result
#     }
# ```
# 
# The above code is copied and pasted into the Lambda code editor and replace with the endpoint name.

# In[45]:


predictor.endpoint


# Once endpoint name added to the Lambda function, click on **Save**. Lambda function is now up and running. Next we need to create a way for our web app to execute the Lambda function.
# 
# ##### Setting up API Gateway
# 
# To create a new API using API Gateway that will trigger the Lambda function created.
# 
# Using AWS Console, navigate to **Amazon API Gateway** and then click on **Get started**.
# 
# On the next page, make sure that **New API** is selected and give the new api a name, for example, `sentiment_analysis_api`. Then, click on **Create API**.
# 
# Now we have created an API, however it doesn't currently do anything. What we want it to do is to trigger the Lambda function that we created earlier.
# 
# Select the **Actions** dropdown menu and click **Create Method**. A new blank method will be created, select its dropdown menu and select **POST**, then click on the check mark beside it.
# 
# For the integration point, make sure that **Lambda Function** is selected and click on the **Use Lambda Proxy integration**. This option makes sure that the data that is sent to the API is then sent directly to the Lambda function with no processing. It also means that the return value must be a proper response object as it will also not be processed by API Gateway.
# 
# Type the name of the Lambda function you created earlier into the **Lambda Function** text entry box and then click on **Save**. Click on **OK** in the pop-up box that then appears, giving permission to API Gateway to invoke the Lambda function you created.
# 
# The last step in creating the API Gateway is to select the **Actions** dropdown and click on **Deploy API**. Create a new Deployment stage and name it, for example `prod`.
# 
# To copy or write down the URL provided to invoke the newly created public API. This URL can be found at the top of the page, highlighted in blue next to the text **Invoke URL**.

# ### 12. Deploying web app
# 
# In the `website` folder there is a file called `index.html`. Replace the string "**\*\*REPLACE WITH PUBLIC API URL\*\***" with the url that earlier on and then save the file.
# 
# Now, if open `index.html` the file, web browser will behave as a local web server to interact with your SageMaker model.

# ##### Delete the endpoint
# 
# Remember to always shut down your endpoint to avoid charging for large bill.

# In[ ]:


predictor.delete_endpoint()

