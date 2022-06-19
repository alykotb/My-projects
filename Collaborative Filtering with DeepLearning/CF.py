#!/usr/bin/env python
# coding: utf-8

# load the necessary modules
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torchmetrics
from torchmetrics import Metric
from torchmetrics import MeanSquaredError, Accuracy, MeanAbsoluteError
from torch import embedding
from torch.utils.data import Dataset, DataLoader, IterableDataset
from sklearn import model_selection, preprocessing, metrics
import random
import matplotlib.pyplot as plt



#extracting ratings from text file into a data frame
df = pd.DataFrame()
# df = pd.read_csv('ratings.csv', encoding='utf-8')
df = pd.read_table('ratings.txt', delimiter = '::')
df.head(3)

#preprocessing

# keeping only column that we want by droping timestamp column
df = df[['user_id','movie_id','rating']]

# Hot-encoding for movies and users to perform vector embeddings
user_label_encoder  = preprocessing.LabelEncoder()
movie_label_encoder = preprocessing.LabelEncoder()
df.user_id  =  user_label_encoder.fit_transform(df.user_id.values)
df.movie_id =  movie_label_encoder.fit_transform(df.movie_id.values)
num_users =  len(user_label_encoder.classes_)
num_movies = len(movie_label_encoder.classes_)


# keeping only column that we want by droping timestamp column
train_ratings = df.sample(frac=0.8) 
test_ratings  = df.drop(train_ratings.index)

train_users  = train_ratings.user_id.values 
train_movies = train_ratings.movie_id.values
train_labels = train_ratings.rating.values 

train_users  = torch.tensor(train_users) 
train_movies = torch.tensor(train_movies) 
train_labels = torch.tensor(train_labels)


# test_ratings  = df.drop(test_ratings.index)
# test_users  = test_ratings.user_id.values 
# test_movies = test_ratings.movie_id.values
# test_labels = test_ratings.rating.values 




vector = user_label_encoder.fit_transform(df.user_id.values)

tensor=torch.tensor(vector)

# defining the dataset class for our model

loss_history = []

class MoviesDataSet(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings
    def __len__(self):
        return len(self.users)
    def __getitem__(self, idx):
        return  self.users[idx] , self.movies[idx],  self.ratings[idx]  
        


# defining our collaborative filtering model

class CollaborativeFiltering(pl.LightningModule):
    def __init__(self, num_users, num_movies, user_training_tensor, movie_training_tensor,
                 label_training_tensor,batch_size):
        super().__init__()
        self.users_tensor = user_training_tensor
        self.movies_tensor = movie_training_tensor
        self.labels_tensor = label_training_tensor
        
        self.user_embedding  = torch.nn.Embedding(num_embeddings = num_users, embedding_dim =32)
        self.movie_embedding = torch.nn.Embedding(num_embeddings = num_movies, embedding_dim =32)
        self.output = torch.nn.Linear(64, 1)
        self.batch_size = batch_size
        
        # defining some metrics attributes
        self.MSE = MeanSquaredError()
        self.MAE = MeanAbsoluteError()
        self.epoch_loss = 0
        loss_history.clear() ##initiate loss history global variable
        
        
    def forward(self, user_input, movie_input):
        user_embedded = self.user_embedding(user_input)
        movie_embedded = self.movie_embedding(movie_input) 
        emb_vector    = torch.cat([user_embedded,movie_embedded], dim=1)
#         print(emb_vector.shape)
        pred = self.output(emb_vector)
        return pred


    def training_step(self, batch, batch_idx):
        user_input, movie_input, labels = batch
        batch_size = len(user_input)
        predicted_labels = self.forward(user_input, movie_input)
        labels = labels[:,None]
        loss = torch.nn.MSELoss()(predicted_labels, labels.view(-1,1).float())
        self.MSE(predicted_labels, labels.view(-1,1).float())
        mse =  self.MSE(predicted_labels, labels.view(-1,1).float())
        mae =  self.MAE(predicted_labels, labels.view(-1,1).float())
        self.log('mean_absolute_error', mae, prog_bar=True)
        self.log('mean_squared_error', mse, prog_bar=True)
        self.log('batch-size', batch_size, prog_bar=True)
        self.log('epoch', self.current_epoch, prog_bar=True)
        epoch = self.current_epoch+1.0
        self.log('epoch', epoch, prog_bar=True)
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
    
    def train_dataloader(self):
        return DataLoader(MoviesDataSet(self.users_tensor, self.movies_tensor, self.labels_tensor),
                          batch_size=self.batch_size)
########## training model class definition ends############################


# utilizing our model to make predictions

####### preparing data to be used in predicitons
moviesIds  = df['movie_id'].unique() #id must be inserted encoded also 
user_item_set = set(zip(df.user_id,df.movie_id)) # we need to use here the full range of ids 
users_interactions = df.groupby('user_id')['movie_id'].apply(list).to_dict()

def get_user_pred_list(user):
#     for i in :
      user_interactions = list(users_interactions[user])
      new_items_to_user = list(set(moviesIds)-set(user_interactions))
      random_selected_new = list(np.random.choice(list(new_items_to_user), 99))
      predicted_labels = np.squeeze(model(torch.tensor([user]*(len(random_selected_new))),
                                   torch.tensor(random_selected_new)).detach().numpy())
      top_5 = [random_selected_new[i] for i in np.argsort(predicted_labels)[::-1][0:5].tolist()]
      predicted_movies_index = np.argsort(predicted_labels)
      print("Movies recommendation list:")
      for i in range(0,5):
        print(f"Movie's id:{top_5[i]+1}")




def parse_args():
    parser = argparse.ArgumentParser(description="Run CF.")
    # parser.add_argument('--path', nargs='?', default='Data/',
    #                     help='Input data path.')
    # parser.add_argument('--dataset', nargs='?', default='ml-1m',
    #                     help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')




    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    model = CollaborativeFiltering(num_users, num_movies, train_users, train_movies, train_labels, batch_size)
    trainer = pl.Trainer(max_epochs=epochs)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    trainer.fit(model)
    ####plot the
    loss = np.array(loss_history)
    plt.plot(loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Cost function")
    plt.show()
    ### After finishing training ###
    while True:
        user = input("Enter the user's id:")
        get_user_pred_list(int(user)-1) ### because the model was trained on dataset in which the users and movies
                                        ### were indexed begining from 0 (where id 1->0)






