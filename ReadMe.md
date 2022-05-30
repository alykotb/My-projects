Complex Representation-Based Link Prediction (Users and Items adjacency matrix)
===============================================================================
This code is an implementation of recommender systems approach presented in the paper, --------------a link------------ prediction approach for item recommendation with complex numbers. Briefly the idea of this approach is based on coverting the directed graph represeting interactions between users and items in a ratings dataset, the similarities whithin items, and the similarities within users into an adjacency matrix (A). Where the interactions between users and items are represented using complex numbers, the like interaction is represented as J while the dislike is represented as -J. While a relation between a user and another user is represented as 1 for being similar while -1 for dissimilar. And the same represntation is applied between an item and another. Then we come out with a matrix consisiting of 4 parts concatentaed together. We have the users-users similarity matrix, the users-items interaction matrix, the items-items similarity matrix, and the items-users similarity one which is the negative transpose of the users-items matrix.

For simplicity,the users-users and items-items are ignored in this approach, so these matrices are set to zero. To predict a relationship (link-prediction) between a user and an item the matrix A is powered to an odd power (because paths length between user and an item is always odd), the resulting powered matrix from the users-items part(matrix) predicts a like or a dislike by a user to an item, where the path length between the user and the item is the value of the odd power. So, the power depends the path's length you want to get information for. After getting the results the hits rate is calculated to evaluate the performance of the prediction algorithm when changing the power and the number of top items (N-top) in  the recommendation list.

Detailed description is found in the paper's link mentioned above.

Terminology
============
Hits rate is the percentage of the most relevant recommended items to users.

Dataset (MovieLens 100k) 
=========================

MovieLens 100k dataset was collected through the MovieLens web site (movielens.umn.edu) during the seven-month period from September 19th, 1997 through April 22nd, 1998. Each user has at least 20 ratings in this dataset. The ratings in this data set are on a 1-5 scale. 

u.data, is a dataset of 100000 ratings by 943 users on 1682 items and the ratings are randomly oredered. Users and items are numbered consecutively from 1. 
This is a tab separated list of user id | item id | rating | timestamp. The time stamps are unix seconds since 1/1/1970 UTC.

In our code we dropped the timestamp data as we need only the interactions between users and items, as well as the ids of both.

GET STARTED
===========
**Run in python environment installed on a machine**

Libraries required to run the code: 

 1. pandas
 2. numpy
 3. argparse (for CF.py only)
 4. sklearn
 5. random


