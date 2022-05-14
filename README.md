Deep Learning Collaborative Filtering
=====================================

This is a readme file for a collaborative filtering deep learning model.

MODEL STRUCTURE
===============
The introduced model performs vector embedding for  users and items that are present in the batch pluged into the model. Where this batch includes
ratings made by users to movies and for each rating the movie's id and the user's id are present in the dataset.
Each user and item in the data set  are represnted into a hot encoded vector of zeros and ones based on its id eg.(user of id 1 is encoded as ([0000.........001])
where the length of the vecotr is the number of unique users in the ratings dataset. The users as hot encoded vecotrs are used by an embedding function
to generate an embedding vectors matrix where each rating  has a vector of weights in this matrix representing its user. The same described process is
done for items generating an embedding vecotrs matrix for the items in the batch. 

After that the two embedding vectors of users and items are concatinated forming the first layer in our learning model. In this layer each rating has 
a concatinated vector of its user and items' embedded vectors.This matrix of user-item vectors corresponding to each rating in the batch pluged as an 
input to a neural network layer consisiting of number of neurons equal to the length of the the user-item vector for each rating. This neural network 
produces an output which is the rating prediction for each item-user vector corresponding to a rating in the dataset.
 

Then using mean square error the loss is calculated for each iteration by comparing the error of the differance between the ratings labels and the predicted ones 
by the model. 


DATASET
=======

MovieLens 1 Million (ml-1m)

The dataset is a set of ratings rated by users to movies.

All ratings are contained in the file "ratings.dat" and are in the
following format:

UserID::MovieID::Rating::Timestamp

- UserIDs range between 1 and 6040 
- MovieIDs range between 1 and 3952
- Ratings are made on a 5-star scale (whole-star ratings only)
- Timestamp is represented in seconds since the epoch as returned by time(2)
- Each user has at least 20 ratings


GET STARTED
===========

Run in python environment installed on a machine:

*If you are using a python environmnet running on your machine not these libraries should be installed (no installation needed in Google Colab):

     1-torch
     2-pandas
     3-numpy
     4-pytorch_lightning
     5-torchmetrics
     6-argparse
     7-sklearn


-Example to run the codes locally in a command-line interpreter:

Run CF.py to train the model:
```
 python CF.py --dataset ml-1m --epochs 20 --batch_size 256 
```


Run the jupyter notebook in Colab:

This video - https://www.youtube.com/watch?v=hAvJN82ulg8 - explains how to upload your dataset file on your google drive
and access it in Google Colab.Then you can run the code normally in the Colab notebook.


-Calculating a recommendation list for a user:

Finding a recoomendation list using CF.py in local environment:

After running the model through the python command-line and the model finishes training the code will keep asking you to enter the
user's id you want to find the top 5 recommenadtion list for a user and whenever it displays a user's list it will ask for the following
user. 

Example:

-Example for Jupyter or Colab notebook:

The last code cell in the notebook file when it is run it will ask you to enter the required user's id and it will provide
with the top 5 recommendation list for the user. 
