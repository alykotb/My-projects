Deep Learning Collaborative Filtering
=====================================

This is a readme file for a collaborative filtering deep learning model.

MODEL STRUCTURE
===============
The introduced model performs vector embedding for users and items that are present in the batch plugged into the model. Where this batch includes ratings made by users for movies and for each rating, the movie's id and the user's id are present in the dataset. Each user and item in the data set are represented into a hot encoded vector of zeros and ones based on its id e.g. (user of id 1 is encoded as ([0000.........001]) where the length of the vector is the number of unique users in the rating dataset. The users as hot encoded vectors are used by an embedding function to generate an embedding vectors matrix where each rating has a vector of weights in this matrix representing its user. The same described process is done for items generating an embedding vectors matrix for the items in the batch.
After that, the two embedding vectors of users and items are concatenated forming the first layer in our learning model. In this layer, each rating has a concatenated vector of its user and items' embedded vectors. This matrix of user-item vectors corresponding to each rating in the batch is plugged as an input to a neural network layer consisting of the number of neurons equal to the length of the user-item vector for each rating. This neural network produces an output which is the rating prediction for each item-user vector corresponding to a rating in the dataset.

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


The time-stamp column in our code is discarded from the datases as the model is based inly on users and movies' ids and the ratings interaction.

GET STARTED
===========
**Run in python environment installed on a machine**

If you are using a python environmnet running on your machine running the CF.py or CF.ipynb not in google-colab these libraries should be installed 
in your environment:

 1. torch
 2. pandas
 3. numpy
 4. pytorch_lightning
 5. torchmetrics 
 6. argparse (for CF.py only)
 7. sklearn
 8. random


Example to train the model and plot the loss locally in a command-line interpreter:

Run CF.py to train the model:
```
 python CF.py --epochs 20 --batch_size 256 
```
