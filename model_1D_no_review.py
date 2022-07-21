import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import pickle
import gzip


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


class Attention(nn.Module):
    def __init__(self, input_size, embed_size, out_channels):
        super(Attention, self).__init__()

        self.input_size = input_size
        self.embed_size = embed_size
        self.out_channels = out_channels
        self.fix_attention_layer = nn.Sequential(
                
        )
        
        self.attention_layer = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(self.input_size, self.embed_size)),
            nn.Sigmoid())
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(5,5)),
            nn.ReLU()
        )
        
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, self.out_channels, kernel_size=(1, self.embed_size)),
            nn.Tanh(),
            nn.MaxPool2d((self.input_size,1)))


    def forward(self, x, senti_scores):
#         x = torch.einsum('ijk,ij->ijk', x, senti_scores)
#         x = x.unsqueeze(1)
#         x = self.cnn(x)
#         global_scores = self.attention_layer(x)
#         out = torch.mul(x, global_scores)
#         out = self.cnn1(out)

        # basic
        # x = x.unsqueeze(1)
        # x = self.cnn1(x)
        
        # basic + senti
#         x = torch.einsum('ijk,ij->ijk', x, senti_scores)
#         x = x.unsqueeze(1)
#         out = self.cnn1(x)
        
        # basic + global
        x = x.unsqueeze(1)
        global_scores = self.attention_layer(x)
        out = torch.mul(x, global_scores)
        out = self.cnn1(out)
        return out

    
class TorchFM(nn.Module):
    def __init__(self, n=600, k=8):
        super().__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.V1 =  nn.Parameter(torch.FloatTensor(n, 1).uniform_(-0.1, 0.1), requires_grad=True)
        self.V2 = nn.Parameter(torch.FloatTensor(n, k).uniform_(-0.1, 0.1), requires_grad=True)
        self.lin = nn.Linear(n, 1)
        self.drop = nn.Dropout(0.5)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.global_bias.data.fill_(0.0)
        
    def forward(self, x):
        one = torch.matmul(x, self.V1)
        out_1 = torch.matmul(x, self.V2)
        out_2 = torch.matmul(x.pow(2), self.V2.pow(2))
        out_3 = 0.5 * (out_1.pow(2) - out_2)
        out_3 = self.drop(out_3)
        out =  out_3.sum(1, keepdim=True)
        
        out = one + out + self.global_bias
        return out    


class SentiAttn(nn.Module):

    def __init__(self, input_size, embed_size=768, channels=400, fc_input_size=200, hidden_size = 500, output_size = 50, latent_v_len = 600, fm_k = 8):
        super(SentiAttn, self).__init__()
        
        self.user_id_dict = pickle.load(gzip.open("data/yelp/user_id_dict.p", 'rb'))
        self.item_id_dict = pickle.load(gzip.open("data/yelp/item_id_dict.p", 'rb'))
        self.num_users = len(self.user_id_dict)
        self.num_items = len(self.item_id_dict)
        self.stddev = 0.1
        self.emb_dim = 100
        self.user_bias = nn.Embedding(self.num_users + 1, 1)
        self.item_bias = nn.Embedding(self.num_items + 1, 1)
        self.user_bias.weight.data.fill_(0.0)
        self.item_bias.weight.data.fill_(0.0)
        
        self.user_emb = nn.Embedding(self.num_users + 1, self.emb_dim)
        self.item_emb = nn.Embedding(self.num_items + 1, self.emb_dim)
        nn.init.normal_(self.user_emb.weight, 0, self.stddev)
        nn.init.normal_(self.item_emb.weight, 0, self.stddev)
        
        self.user_pos_Attention = Attention(input_size, embed_size, channels)
        self.user_neg_Attention = Attention(input_size, embed_size, channels)
        self.item_pos_Attention = Attention(input_size, embed_size, channels)
        self.item_neg_Attention = Attention(input_size, embed_size, channels)
        
        
        self.user_item_Attention = Attention(input_size*4, embed_size, channels)
        self.fcLayer = nn.Sequential(
            nn.ReLU(),
        )
        self.fmlayer = TorchFM(latent_v_len, fm_k)


    def forward(self, user_id, item_id, x_user_pos, x_user_neg, x_item_pos, x_item_neg, user_pos_senti, user_neg_senti, item_pos_senti, item_neg_senti):
        
        user_id = to_var(torch.from_numpy(np.array([self.user_id_dict[uid] for uid in user_id])))
        item_id = to_var(torch.from_numpy(np.array([self.item_id_dict[iid] for iid in item_id])))
        
        x = torch.cat([x_user_pos, x_user_neg, x_item_pos, x_item_neg], 1)
        x_ones = torch.ones(x.size())
        x_senti = torch.cat([user_pos_senti, user_neg_senti, item_pos_senti, item_neg_senti], 1)
        # idx = torch.randperm(x.size(1))
        # shuf_x = x[:,idx]
        
        user_item = self.user_item_Attention(x, x_senti)
        user_item = self.fcLayer(user_item)
        user_item = user_item.squeeze()
        fm_k = 8
        user_emb = self.user_emb(user_id)
        item_emb = self.item_emb(item_id)
        # print("user_item: ", user_item.size())
        z = torch.cat([user_item, user_emb, item_emb], 1)
        
        #print(z.size())
        out =self.fmlayer(z) + self.user_bias(user_id) + self.item_bias(item_id)
        
        return out
