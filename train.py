import torch
import torch.nn as nn
import data_loader
import model_1D_no_review
import numpy as np
import pandas as pd
import pickle
import time
from datetime import datetime
from torch.autograd import Variable
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

def evaluation(target,cf_out):
    fpr, tpr, _ = metrics.roc_curve(target, cf_out)
    auc = metrics.auc(fpr, tpr)
    return auc


def to_var(x, volatile=False):
    # This is not necessary for PyTorch version after 0.4.0
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


# Hyper Parameters
input_size = 20
num_epochs = 20
batch_size = 2048
learning_rate = 1e-4


print("FM 32b no linear attention learning rate 1e-4 yelp dataset v1")
print("Loading data...")

train_loader = data_loader.get_loader('addSenti_train.p', batch_size)
val_loader = data_loader.get_loader('addSenti_valid.p', batch_size, shuffle=False)
test_loader = data_loader.get_loader('addSenti_test.p', batch_size, shuffle=False)

print("train/val/test/: {:d}/{:d}/{:d}".format(len(train_loader), len(val_loader),len(test_loader)))
print("==================================================================================")

#time.sleep(1000)

SentiAttn = model_1D_no_review.SentiAttn(input_size)
# print(SentiAttn)


if torch.cuda.is_available():
    SentiAttn.cuda()

#time.sleep(1000)
# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(SentiAttn.parameters(), lr=learning_rate)

print("==================================================================================")
print("Training Start..")

batch_loss = 0
# Train the Model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (user_id, item_id, user_pos, user_neg, item_pos, item_neg, user_pos_senti, user_neg_senti, item_pos_senti, item_neg_senti, labels) in enumerate(train_loader):

        
        # Convert torch tensor to Variable
        batch_size = len(user_pos)
        user_id = list(user_id)
        item_id = list(item_id)
        user_pos = to_var(user_pos)
        user_neg = to_var(user_neg)
        item_pos = to_var(item_pos)
        item_neg = to_var(item_neg)
        user_pos_senti = to_var(user_pos_senti)
        user_neg_senti = to_var(user_neg_senti)
        item_pos_senti = to_var(item_pos_senti)
        item_neg_senti = to_var(item_neg_senti)
        labels = to_var(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = SentiAttn(user_id, item_id, user_pos, user_neg, item_pos, item_neg, user_pos_senti, user_neg_senti, item_pos_senti, item_neg_senti)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        batch_loss += loss.data
        #time.sleep()
        if i%100 ==0:
            # Print log info
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f '
                  % (epoch, num_epochs, i, total_step,
                      batch_loss/100, np.exp(loss.data.cpu())), datetime.now())
            batch_loss = 0
    # Save the Model
    torch.save(SentiAttn.state_dict(), 'result/abla_result_no_review/model_yelp_'+str(epoch)+'.pkl')
    print("==================================================================================")
    print("Testing Start..")

    for i, (user_id, item_id, user_pos, user_neg, item_pos, item_neg, user_pos_senti, user_neg_senti, item_pos_senti, item_neg_senti, labels) in enumerate(test_loader):

        # Convert torch tensor to Variable
        batch_size = len(user_pos)
        user_id = list(user_id)
        item_id = list(item_id)
        user_pos = to_var(user_pos)
        user_neg = to_var(user_neg)
        item_pos = to_var(item_pos)
        item_neg = to_var(item_neg)
        user_pos_senti = to_var(user_pos_senti)
        user_neg_senti = to_var(user_neg_senti)
        item_pos_senti = to_var(item_pos_senti)
        item_neg_senti = to_var(item_neg_senti)
        labels = to_var(labels)

        outputs = SentiAttn(user_id, item_id, user_pos, user_neg, item_pos, item_neg, user_pos_senti, user_neg_senti, item_pos_senti, item_neg_senti)
        if i == 0:
            result = outputs.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            target = np.reshape(labels, len(labels))
        else:
            result = np.append(result, outputs.data.cpu().numpy(),axis=0)
            labels = labels.data.cpu().numpy()
            target = np.append(target, np.reshape(labels, len(labels)), axis=0)
        if i % 100 == 0:
            print("Ith round: ", i)
    print("Test Performance of Epoch: ", epoch)
    print("mae: ", mean_absolute_error(target, result))
    print("rmse: ", sqrt(mean_squared_error(target, result)))
    pickle.dump((target, result),open('result/abla_result_no_review/result_test'+ str(epoch) + 'yelp.pickle','wb'))

    #pickle.dump(result,open('result_val.pickle','wb'))

    print("==================================================================================")
    print("Testing End..")
print("==================================================================================")
print("Training End..")

# print("==================================================================================")
# print("Testing Start..")

# for i, (user_pos, user_neg, item_pos, item_neg, user_pos_senti, user_neg_senti, item_pos_senti, item_neg_senti, labels) in enumerate(val_loader):
    
#     # Convert torch tensor to Variable
#     batch_size = len(user_pos)
#     user_pos = to_var(user_pos)
#     user_neg = to_var(user_neg)
#     item_pos = to_var(item_pos)
#     item_neg = to_var(item_neg)
#     user_pos_senti = to_var(user_pos_senti)
#     user_neg_senti = to_var(user_neg_senti)
#     item_pos_senti = to_var(item_pos_senti)
#     item_neg_senti = to_var(item_neg_senti)
#     labels = to_var(labels)

#     outputs = SentiAttn(user_pos, user_neg, item_pos, item_neg, user_pos_senti, user_neg_senti, item_pos_senti, item_neg_senti)
#     if i == 0:
#         result = outputs.data.cpu().numpy()
#         labels = labels.tolist()
#         target = np.reshape(labels, len(labels))
#     else:
#         result = np.append(result, outputs.data.cpu().numpy(),axis=0)
#         labels = labels.tolist()
#         target = np.append(target, np.reshape(labels, len(labels)), axis=0)
#     if i % 100 == 0:
#         print("Ith round: ", i)
        
# pickle.dump((target, result),open('result_test_fm_100e_32b_1e-4l_yelp_v3.pickle','wb'))


#pickle.dump(result,open('result_val.pickle','wb'))

print("==================================================================================")
print("Testing End..")
