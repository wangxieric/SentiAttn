import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import pickle
import gzip
import numpy as np
from datetime import datetime 

def preprocess_review(reviews, scores):
        reviews_emb = {}
        processed_scores = {}
        print("number of users: ", len(reviews))
        count = 0
        for u in reviews.keys():
            # count number of reviews and their associcated score
            review = reviews[u]
            score = scores[u]
            
            review_len = len(review)
            score_len = len(score)

            #reviews = reviews.keys()
            if review_len > 20:
                review = review[:20]

            if score_len > 20:
                score = score[:20]

            total_review = np.array([])

            for i, review_str in enumerate(reviews[u]):
                input_ids = torch.tensor(tokenizer.encode(review_str[:512])).unsqueeze(0).to('cuda')
                output = model(input_ids)[1]
                if i == 0:
                    review_vec = output.data.cpu().numpy()
                else:
                    review_vec = np.concatenate([review_vec, output.data.cpu().numpy()], axis=0)
                
            score = np.array(score)
            
            if review_len < 20:
                pad_len = 20 - review_len
                total_scores =  np.concatenate((score, np.asarray([0.0] * pad_len)))
                pad_vector = np.zeros((pad_len, 768))
                review_vec = np.concatenate((review_vec, pad_vector), axis=0)
            
            reviews_emb[u] = review_vec
            processed_scores[u] = total_scores
            if count % 1000 == 0:
                print(count, datetime.now())
            count += 1
        return reviews_emb, processed_scores

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to('cuda')

print("neg item amazon")
root='../DeepSenti/data/'

# pos_user_reviews = pickle.load(gzip.open(root + 'yelp/pos_user_reviews.p', 'rb'))
# neg_user_reviews = pickle.load(gzip.open(root + 'yelp/neg_user_reviews.p', 'rb'))
# pos_item_reviews = pickle.load(gzip.open(root + 'yelp/pos_item_reviews.p', 'rb'))
# neg_item_reviews = pickle.load(gzip.open(root + 'yelp/neg_item_reviews.p', 'rb'))

# pickle user and item sentiment scores of Yelp dataset
# pos_user_senti = pickle.load(gzip.open(root + 'yelp/pos_user_senti_v2.p', 'rb'))
# neg_user_senti = pickle.load(gzip.open(root + 'yelp/neg_user_senti_v2.p', 'rb'))
# pos_item_senti = pickle.load(gzip.open(root + 'yelp/pos_item_senti_v2.p', 'rb'))
# neg_item_senti = pickle.load(gzip.open(root + 'yelp/neg_item_senti_v2.p', 'rb'))


# pos_user_reviews = pickle.load(gzip.open(root + 'amazon/pos_user_reviews.p', 'rb'))
# neg_user_reviews = pickle.load(gzip.open(root + 'amazon/neg_user_reviews.p', 'rb'))
# pos_item_reviews = pickle.load(gzip.open(root + 'amazon/pos_item_reviews.p', 'rb'))
neg_item_reviews = pickle.load(gzip.open(root + 'amazon/neg_item_reviews.p', 'rb'))

# pickle user and item sentiment scores of Amazon dataset
# pos_user_senti = pickle.load(gzip.open(root + 'amazon/pos_user_senti_v2.p', 'rb'))
# neg_user_senti = pickle.load(gzip.open(root + 'amazon/neg_user_senti_v2.p', 'rb'))
# pos_item_senti = pickle.load(gzip.open(root + 'amazon/pos_item_senti_v2.p', 'rb'))
neg_item_senti = pickle.load(gzip.open(root + 'amazon/neg_item_senti_v2.p', 'rb'))


# pos_user_review_emb, pos_user_senti_processed = preprocess_review(pos_user_reviews, pos_user_senti)
# pickle.dump([pos_user_review_emb, pos_user_senti_processed], gzip.open('data/yelp/pos_user.p', 'wb'))

# neg_user_review_emb, neg_user_senti_processed = preprocess_review(neg_user_reviews, neg_user_senti)
# pickle.dump([neg_user_review_emb, neg_user_senti_processed], gzip.open('data/yelp/neg_user.p', 'wb'))

# pos_item_review_emb, pos_item_senti_processed = preprocess_review(pos_item_reviews, pos_item_senti)
# pickle.dump([pos_item_review_emb, pos_item_senti_processed], gzip.open('data/yelp/pos_item.p', 'wb'))

# neg_item_review_emb, neg_item_senti_processed = preprocess_review(neg_item_reviews, neg_item_senti)
# pickle.dump([neg_item_review_emb, neg_item_senti_processed], gzip.open('data/yelp/neg_item.p', 'wb'))

# pos_user_review_emb, pos_user_senti_processed = preprocess_review(pos_user_reviews, pos_user_senti)
# pickle.dump([pos_user_review_emb, pos_user_senti_processed], gzip.open('data/amazon/pos_user.p', 'wb'))

# neg_user_review_emb, neg_user_senti_processed = preprocess_review(neg_user_reviews, neg_user_senti)
# pickle.dump([neg_user_review_emb, neg_user_senti_processed], gzip.open('data/amazon/neg_user.p', 'wb'))

# pos_item_review_emb, pos_item_senti_processed = preprocess_review(pos_item_reviews, pos_item_senti)
# pickle.dump([pos_item_review_emb, pos_item_senti_processed], gzip.open('data/amazon/pos_item.p', 'wb'))

neg_item_review_emb, neg_item_senti_processed = preprocess_review(neg_item_reviews, neg_item_senti)
pickle.dump([neg_item_review_emb, neg_item_senti_processed], gzip.open('data/amazon/neg_item.p', 'wb'))





