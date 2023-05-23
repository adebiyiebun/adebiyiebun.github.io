#!/usr/bin/env python
# coding: utf-8
import torch
import flickr_model
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def collate(scores):   
    # calculate median and MAD
    med = np.median(scores)
    mad = np.median(np.abs(scores - med))

    # define threshold for outliers
    threshold = 2 * mad

    # remove outliers
    filtered_scores = scores[np.abs(scores - med) < threshold]

    # calculate mean of remaining scores
    mean_score = np.mean(filtered_scores)
    return mean_score

class ImageAPR():
    def __init__(self,model_file = 'apr_20.pt'):
        self.model = torch.load(model_file)
        

    def predict(self, X,agg):
#         for image in X:
        X = np.array(X).astype('float32')
        X = torch.tensor(X)
        X = Variable(X.cuda())
        user_pr,_ = self.model(X)
        predictions = user_pr.cpu().detach().numpy()
        if agg == True:
            user = []
            for i in range(0,5):
                pred = predictions[:, i]
                mean = collate(pred)
                user.append(mean)
            return user
        else:
            return predictions
            
    
    def score(self,X_test,y_test):
        y_pred = self.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        return mae, rmse
    

class PARS():
    def __init__(self,user_id,user,i_csv_file,user_data):
        self.uid = user_id
        self.tweets = user['tweets']
        self.features = user['images']
        self.textapr = user['textapr']
        self.imageapr =user['imageapr']
        self.mmapr = user['mmapr']
        self.imagedata = pd.read_csv(i_csv_file)
        self.image_apr = ImageAPR()
        self.userdata =user_data
       
        self.recommendations =[]
        self.index =0
 
    def userdata(file):
        with open(file,'rb') as f:
            x = pickle.load(x,f)
        return x
    
    def select_apr(self,apr):
        if apr == 'text':
            self.method = 'textapr'
            return self.textapr
        elif apr == 'image':
            self.method = 'imageapr'
            return self.imageapr
        else:
            self.method ='mmapr'
            return self.mmapr
        
    def image_apr(self,features):
        predictions = self.image_apr.predict(features ,agg=False)
        return predictions
    
    def get_recommendations(self,apr):
        user_personality = self.select_apr(apr)
        images = self.rank_images(self.userdata, self.uid,user_personality, 10)
        paths  = [t[-1] for t in images]
        self.recommendation = paths

        
    def return_recs(self,apr):
        self.get_recommendations(apr)
        y = self.recommendations[self.index:self.index+15]
        self.index = self.index+16
        return y      

    def compute_neighbourhood(self,target_user, user_personality, n):
        similarities = {}
        target = user_personality
        target = fill_nans_infs(target, fill_value=0.5)

        for user, data in self.userdata.items():
            if user != self.uid:
                x = data[self.method]
                x = fill_nans_infs(x, fill_value=0.5)
        
                similarity = pearsonr(target, x)[0]
                similarities[user] = similarity

        similar_users = sorted(similarities, key=similarities.get, reverse=True)[:n]
        return similar_users

    def rank_images(self,users, target_user,user_personality, n):
        similar_users = self.compute_neighbourhood( target_user,user_personality, n)
        user_personality = np.array(user_personality).reshape(1, -1)
        image_similarities = {}
        
        for user in similar_users:
            images = users[user]['images']
            image_ids = users[user]['image_id']

            processed_images = self.image_apr.predict(images,agg=False)
            similarity = cosine_similarity(user_personality, processed_images)

            for i in range(len(image_ids)):
                image_similarities[(user, i)] = (similarity[0][i], image_ids[i])

        ranked_images = sorted(image_similarities, key=lambda x: image_similarities[x][0], reverse=True)
        ranked_image_ids = [(user, i, image_similarities[(user, i)][1]) for user, i in ranked_images]
        return ranked_image_ids
    
    




        
        
        

with open('pars_data.pickle','rb') as f:
    y = pickle.load(f)
y


def fill_nans_infs(array, fill_value):
    # Fill NaN and Inf values with the specified fill_value
    filled_array = np.nan_to_num(array, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return filled_array



pars = PARS('-eugenia-',y['-eugenia-'],'dataset/image_data.csv',y)
text = pars.get_recommendations('text')
image = pars.get_recommendations('image')
mm = pars.get_recommendations('mm')





