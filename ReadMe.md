Complex Representation-Based Link Prediction (Users and Items Adjacency Matrix)
===============================================================================
This code is an implementation of the recommender systems approach presented in the paper, [A link prediction approach for item recommendation with complex numbers](https://www.sciencedirect.com/science/article/pii/S0950705115000568). Briefly, the idea of this approach is based on converting the directed graph representing interactions between users and items in a rating dataset, the similarities within items, and the similarities within users into an adjacency matrix (A). Where the interactions between users and items are represented using complex numbers, the like interaction is represented as J while the dislike is represented as -J. While a relationship between a user and another user is represented as 1 for being similar while -1 for dissimilar. And the same representation is applied between one item and another. Then we come out with a matrix consisting of 4 parts concatenated together. We have the users-users similarity matrix, the users-items interaction matrix, the items-items similarity matrix, and the items-users similarity one which is the negative transpose of the users-items matrix.

For simplicity, the users-users and items-items are ignored in this approach, so these matrices are set to zero. To predict a relationship (link-prediction) between a user and an item the matrix A is powered to an odd power (because the length of the paths between a user and an item is always odd), and the resulting powered matrix from the users-items part(matrix) predicts a like or a dislike by a user to an item, where the path length between the user and the item is the value of the odd power. So, the power depends on the path's length you want to get information for. After getting the results the hits rate is calculated to evaluate the performance of the prediction algorithm when changing the power and the number of top items (N-top) in the recommendation list.

The detailed description is found in the paper's link mentioned above


Terminology
============
Hits rate is the percentage of the most relevant recommended items to users.

Dataset (MovieLens 100k) 
=========================
MovieLens 100k dataset was collected through the MovieLens website (movielens.umn.edu) during the seven-month period from September 19th, 1997 through April 22nd, 1998. Each user has at least 20 ratings in this dataset. The ratings in this data set are on a 1-5 scale.

u.data is a dataset of 100000 ratings by 943 users on 1682 items and the ratings are randomly ordered. Users and items are numbered consecutively from 1. This is a tab-separated list of user id | item id | rating | timestamp. The timestamps are Unix seconds since 1/1/1970 UTC.

In our code, we dropped the timestamp data as we need only the interactions between users and items, as well as the ids of both.


GET STARTED
===========

**Libraries required to run the code** 
 1. pandas
 2. numpy
 3. sklearn
 4. random

**Run the notebook, CORLP.ipynb in a local python environment using Jupyter Notebook**

You have to install the above mentioned libraries in your Jupyter Notebook's project.

Run the second notebook cell to read the dataset's file whe running the code locally:
```
df = pd.DataFrame()
path  = "u.data"
df = pd.read_csv(path, sep="\t", header=None, names=['user_id','movie_id','rating','timestamp'])
df.head()
```

**Running CORLP.ipynb using Google Colab**

This video - https://www.youtube.com/watch?v=hAvJN82ulg8 - explains how to upload your dataset file on your google drive
and access it in Google Colab. 

Then, run the third notebook cell to read the dataset's file in Colab:
```
df = pd.DataFrame()
path  = "/content/drive/MyDrive/u.data"
df = pd.read_csv(path, sep="\t", header=None, names=['user_id','movie_id','rating','timestamp'])
df.head()
```

**For both Colab and Jupyter Notebook**

Beside running the cell of reading the datasetfile base od your platform, run all of the notebook's cells.

In the last cell of the notebook enter the path-length and the top-N list's length to get the hits rate corresponding to them:
```
path_length = input("Enter the path length:")
n = input("Enter the length of top-N items:")
Evaluate_Model(int(n), int(path_length)
```



