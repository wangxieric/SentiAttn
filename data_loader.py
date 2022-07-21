import torch
from torch.utils import data
import pandas as pd
import pickle
import gzip
import numpy as np
import os
from nltk import word_tokenize
from torch.utils.data.dataloader import default_collate
from datetime import datetime

class ReviewDataset(data.Dataset):


    def __init__(self,pickle_path,root='data/amazon/',max_len = 10000):

        #self.dataset = pd.read_csv(root+csv_path)
        # self.dataset = pickle.load(gzip.open("../DeepSenti/"+ root + pickle_path,'rb'))
        self.dataset = pickle.load(gzip.open("../office_products/"+ pickle_path,'rb'))
        self.dataset.reset_index()        
        print("Loading pre-trained embedding...")
        self.pos_user_review, self.pos_user_senti = pickle.load(gzip.open(root + 'pos_user.p', 'rb'))
        self.neg_user_review, self.neg_user_senti = pickle.load(gzip.open(root + 'neg_user.p', 'rb'))
        self.pos_item_review, self.pos_item_senti = pickle.load(gzip.open(root + 'pos_item.p', 'rb'))
        self.neg_item_review, self.neg_item_senti = pickle.load(gzip.open(root + 'neg_item.p', 'rb'))
        print("Finish loading pre-trained embedding...")
        
    def __getitem__(self, index):
        #print(index)
        row = self.dataset.iloc[index]
        #print(row.shape)
        user_id = row['user_id']
        item_id = row['business_id']
        
        # print("user_id: ", user_id)
        # the max length is 20 and the length of sentiment score vector can be shorter than 20
        pos_user_review, pos_user_senti = (torch.from_numpy(self.pos_user_review[user_id][:20]).float(), 
                                           torch.from_numpy(self.pad_len(self.pos_user_senti[user_id])).float())
        neg_user_review, neg_user_senti = (torch.from_numpy(self.neg_user_review[user_id][:20]).float(),
                                           torch.from_numpy(self.pad_len(self.neg_user_senti[user_id])).float())
        pos_item_review, pos_item_senti = (torch.from_numpy(self.pos_item_review[item_id][:20]).float(), 
                                           torch.from_numpy(self.pad_len(self.pos_item_senti[item_id])).float())
        neg_item_review, neg_item_senti = (torch.from_numpy(self.neg_item_review[item_id][:20]).float(), 
                                           torch.from_numpy(self.pad_len(self.neg_item_senti[item_id])).float())
        
        # print("pos_user_review: ", pos_user_review.size())
        # print("pos_user_senti: ", pos_user_senti.size())

        #rating 
        # target = torch.Tensor([row['stars']]).float()
        target = torch.FloatTensor([row['stars'].astype(np.float64)])
    
        return (user_id, item_id, pos_user_review, neg_user_review, pos_item_review, neg_item_review, 
                pos_user_senti, neg_user_senti, pos_item_senti, neg_item_senti, target)

    
    def pad_len(self, senti, max_len=20):
        if len(senti) < max_len:
            pad_vec = np.array([0.0] * (max_len - len(senti)))
            return np.append(senti, pad_vec)
        return senti
    
    
    def __len__(self):
        """Returns the total number of image files."""
        return len(self.dataset)

    
    def preprocess_review(self,reviews, scores):
    
        review_len = len(reviews)
        score_len = len(scores)
        
        #reviews = reviews.keys()
        if review_len > 100:
            reviews = reviews[:100]
        
        if score_len > 100:
            scores = scores[:100]

        total_review = np.array([])
        total_scores = np.array([])
        
        for i, review_str in enumerate(reviews):
            review = word_tokenize(review_str)
            l =  len(review)
            total_review = np.concatenate((total_review, review))
            total_scores = np.concatenate((total_scores, np.asarray([scores[i]] * l)))
            total_review = np.append(total_review, '+++')
            total_scores = np.append(total_scores, 0.0)
            

        review = []
        for word in total_review:
            if word == '+++':
                review.append(self.delimiter)
            else:
                if word in self.glove:
                    review.append(self.glove[word])
                else:
                    review.append(self.unknown)
        
        # This review is word embedding converted review text
        review = np.array(review)

        if len(review) < self.max_len:
            pad_len = self.max_len - len(review)
            total_scores =  np.concatenate((total_scores, np.asarray([0.0] * pad_len)))
            pad_vector = np.zeros((pad_len,100))
            review = np.concatenate((review, pad_vector), axis=0)
        else:
            review = review[:self.max_len]
            total_scores = total_scores[:self.max_len]
        #print(total_scores)
        return review, total_scores

        
        
def my_collate(batch):

    batch = list(filter(lambda x:x is not None, batch))
    return default_collate(batch)


def get_loader(data_path, batch_size=100, shuffle=True, num_workers=0):
    """Builds and returns Dataloader."""
    
    dataset = ReviewDataset(data_path)
    #print(len(dataset))
    data_loader = data.DataLoader(dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=my_collate)
    return data_loader
