#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 16:19:04 2019
TWITTER SENTIMENT ANALYSIS
@author: rajat

Recommender system

1.INstall dependencies
2.Write script
"""
import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# fetch data and format it
data = fetch_movielens(min_rating=4.2)

# printing train and testing data
print('Printing training Data:')
print(repr(data['train']))

print('Testing Data:')
print(repr(data['test']))

# create model

model = LightFM(learning_rate=0.02,loss='warp')
 
# train model
model.fit(data['train'], epochs=30, num_threads=5)

def sample_recommendation(model, data, user_ids):
    # number of users nd movies
    n_users, n_items = data['train'].shape
    
    # generate recommmendation for each user
    for userid in user_ids:
        # movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[userid].indices]
        
        # movies our model predict they will like
        scores = model.predict(userid, np.arange(n_items))
        
        top_items = data['item_labels'][np.argsort(-scores)]
        
        print("User %s"% userid)
        print("   Known positives:")
        for x in known_positives[:3]:
            print("   %s"%x)
        print("   Recommendations:")
        for x in top_items[:3]:
            print("   %s"%x)     
            
sample_recommendation(model, data,[3,15,42])            