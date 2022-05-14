


GET STARTED
===========

-Run in python environment installed on a machine:

If you are using a python environmnet running on your machine not google-colab these libraries should be installed:

 1-torch
 2-pandas
 3-numpy
 4-pytorch_lightning
 5-torchmetrics
 6-argparse
 7-sklearn


Example to run the codes locally in a command-line interpreter:

Run CF.py to train the model:
```
 python CF.py --dataset ml-1m --epochs 20 --batch_size 256 
```


Run the jupyter notebook in Colab:

This video - https://www.youtube.com/watch?v=hAvJN82ulg8 - explains how to upload your dataset file on your google drive
and access it in Google Colab.Then you can run the code normally in the Colab notebook.




Calculating a recommendation list for a user:

-Finding a recoomendation list using CF.py in local environment:

After running the model through the python command-line and the model finishes training the code will keep asking you to enter the
user's id you want to find the top 5 recommenadtion list for a user and whenever it displays a user's list it will ask for the following
user. 

Example:



-Example for Jupyter or Colab notebook:

The last code cell in the notebook file when it is run it will ask you to enter the required user's id and it will provide
with the top 5 recommendation list for the user. 
