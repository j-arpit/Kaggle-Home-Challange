#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
import time


# ### Import Pre-processed Data

# In[2]:


X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')
test = np.load('test.npy')


# In[3]:


##Import libraries for neural network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ### Prepare data for pytorch

# In[4]:


X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
training = TensorDataset(X_train, y_train)

X_val = torch.Tensor(X_val)
y_val = torch.Tensor(y_val)
validation = TensorDataset(X_val, y_val)

test = torch.Tensor(test) ## As test doesnot have target variable so there's no need to create Dataset Here

print("{}, {}, {}, {}, {}".format(X_train.dtype, X_val.dtype, y_train.dtype, y_val.dtype, test.dtype))


# In[14]:


trainset = DataLoader(training, batch_size=10, shuffle=True)
validationset = DataLoader(validation, batch_size=10)
testset = DataLoader(test, batch_size=10)


# ### Neural Network

# In[6]:


class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(133, 64) # input, output for a layer
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 16) 
        self.fc5 = nn.Linear(16, 1)# 1 Target Variable
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        
        return x

net = Net()
print(net)


# ### Training 

# In[8]:


optimizer = optim.Adam(net.parameters(), lr=0.001)
EPOCHS = 100
start = time.time()
for epoch in range(EPOCHS):
    for data in trainset:
        X,y = data
        net.zero_grad()
        output = net(X)
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)
end = time.time()
print(end - start)


# ### Validation

# In[11]:


correct = 0
total = 0
with torch.no_grad():
    for data in validationset:
        X,y = data
        output = net(X)
        print(F.mse_loss(output, y))


# ### Time for Prediction

# In[15]:


pred = torch.LongTensor()
for data in testset:
    X = data
    output = net(X)
    pred = torch.cat((pred, output), dim=0)


# In[16]:


pred.shape


# ### Submission For Kaggle

# In[20]:


submission = pred.detach().numpy()
submission = np.expm1(submission)
submission_file = pd.read_csv("sample_submission.csv")
submission_file.iloc[:, 1] = np.floor(submission)
submission_file.to_csv("House_price_submission_nn.csv", index=False)

