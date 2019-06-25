import json
import pickle
import random

import torch
from torch import nn, optim
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils

from logger import Logger


import torch
torch.cuda.set_device(1)

print('import over')


class FCNN(nn.Module):
    def __init__(self, use_cuda, in_dim, hidden1_dim, hidden2_dim):
        super(FCNN,self).__init__()
        
        self.use_cuda = use_cuda
        
        self.layer1=nn.Linear(in_dim, hidden1_dim)
        self.layer2=nn.Linear(hidden1_dim, hidden2_dim)
        self.layer3=nn.Linear(hidden2_dim, 1)
        
        self.activate_fun=nn.ReLU()#.Sigmoid()
        self.softmax=nn.Softmax()
        self.dropout=nn.Dropout(0.5)
        
        self.loss_func = nn.MSELoss()
        
    def forward(self,x):
        
        if self.use_cuda:
            x=x.cuda()
        
        x1 = self.layer1(x)
        #x1 = self.dropout(x1)
        x1 = self.activate_fun(x1)
        
        x2 = self.layer2(x1)
        #x2 = self.dropout(x2)
        x2 = self.activate_fun(x2)
        
        y = self.layer3(x2)
        return y
    
    def get_loss(self, predicts, labels):
        if self.use_cuda:
            labels = labels.cuda()
        return self.loss_func(predicts, labels)
    
#train set, generate data randomly.
import math
def generate_data_func(x):
    return math.sin(x[0]*x[2])-math.cos(x[0]+x[1]*x[3]-x[4])+x[1]**2+x[1]*x[4]+math.exp(x[5]*x[0]-x[3])

num = 100000
inputs=[]
labels=[]
for ii in range(num):
    x = [random.random() for jj in range(6)]
    label = generate_data_func(x)
    label = (1 + random.random()*0.05)*generate_data_func(x)
    labels.append(label)
    inputs.append(x)
    
inputs = Variable(torch.FloatTensor(inputs))
labels = Variable(torch.FloatTensor(labels))



# train
in_dim= 6
hidden1_dim=4
hidden2_dim=2
use_cuda = 1

logger = Logger('./logs')


fcnn_model = FCNN(use_cuda=use_cuda, 
            in_dim= in_dim, 
            hidden1_dim=hidden1_dim, 
            hidden2_dim=hidden2_dim
           )
if use_cuda:
    fcnn_model.cuda()
    
#adam optimizer
lr=0.0001
optimizer = optim.Adam(filter(lambda p:p.requires_grad, fcnn_model.parameters()),lr = lr)

all_loss=[]
for epoch in range(10000):
    predicts = fcnn_model(inputs)
    loss = fcnn_model.get_loss(predicts, labels)
    #
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    all_loss.append(loss.data[0])
    
    logger.scalar_summary('loss', loss.data[0], epoch)
    
print(len(all_loss))