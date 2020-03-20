import sys
import pickle
import torch
import torch.nn as nn
import numpy as np
from os.path import join
from model import ConvNet

# variables
subject_num = sys.argv[1]
data_path = "./data"
batch_size = 512
epoch = 30

# for debug
# np.random.seed(0)

# divide data into minibatches
def minibatch(data, batch_size):
    start = 0
    while True:

        end = start + batch_size
        yield data[start:end]

        start = end
        if start >= len(data):
            break

# calculate acc
def cal_acc(pred, target):
    assert len(pred) == len(target)
    acc = np.sum(pred == target) / len(pred)
    return acc

def cal_f(pred, target):
    assert len(pred) == len(target)
    tp = 0
    for i in range(len(pred)):
        if pred[i] == target[i] and pred[i] == 1:
            tp += 1
    percision = tp / np.sum(pred == 1)
    recall = tp / np.sum(target == 1)
    f_score = (2 * percision * recall) / (percision + recall)
    return f_score, percision, recall

# train function
def train_batch(model, criterion, optimizer, batch):

    model.zero_grad()

    # forward pass
    x = torch.FloatTensor([i for i in batch[:, 0]]).cuda()
    _, height, width = x.size()
    x = x.view(min(batch_size, len(x)), 1, height, width)
    y = torch.FloatTensor([i for i in batch[:, 1]]).cuda()
    pred = model(x)

    # back proporgation
    loss = criterion(pred.view(-1), y)
    loss.backward()
    optimizer.step()

    pred = pred.cpu().detach().numpy().reshape(-1)
    pred = np.array([1 if n >= 0.5 else 0 for n in pred])
    return pred

def val_batch(model, criterion, optimizer, batch):
    
    with torch.no_grad():
        
        # forward pass
        x = torch.FloatTensor([i for i in batch[:, 0]]).cuda()
        _, height, width = x.size()
        x = x.view(min(batch_size, len(x)), 1, height, width)
        y = torch.FloatTensor([i for i in batch[:, 1]]).cuda()
        pred = model(x)
        
        pred = pred.cpu().detach().numpy().reshape(-1)
        pred = np.array([1 if n >= 0.5 else 0 for n in pred])
        return pred
    
####### main logic #######
# load the data
# data format: [(x, y, y_stim)]
with open(join(data_path, f"s{subject_num}.pkl"), "rb") as infile:
    data = pickle.load(infile)
data_size = len(data)

# shuffle data
shuffle_idx = np.random.permutation(data_size)
data = data[shuffle_idx]

# 80-20 split train/test
cutoff = int(data_size * 80 // 100)
train_data = data[:cutoff]
test_data = data[cutoff:]

# balance label in the train_data
train_data_size = len(train_data)
train_data_true_count = np.sum([x[1] for x in train_data])
train_data_false_count = train_data_size - train_data_true_count 

assert train_data_false_count >= train_data_true_count 

train_data_dup_count = train_data_false_count - train_data_true_count 
train_data_true_idx = np.array([i for i, x in enumerate(train_data) if x[1] == 1])
train_data_true_sample_idx = np.random.choice(train_data_true_idx, train_data_dup_count, replace=True)
train_data_addon = train_data[train_data_true_sample_idx]

# make sure that all the addon have true labels
assert all([x[1] == 1 for x in train_data_addon])

# stack the addon to the original trainning data and shuffle again
train_data = np.concatenate((train_data, train_data_addon), axis=0)
train_data_size = len(train_data)
shuffle_idx = np.random.permutation(train_data_size)
train_data = train_data[shuffle_idx]

# init model
model = ConvNet()
model = model.cuda()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5 * 2, weight_decay=1e-2)

# train loop
# use k-fold validation
k_fold = 10
fold_size = int(train_data_size // k_fold)
for i in range(k_fold):

    # split data into train/val
    val_data_curr_fold = train_data[i*fold_size:(i+1)*fold_size]
    train_data_curr_fold_head = train_data[:i*fold_size]
    train_data_curr_fold_tail = train_data[(i+1)*fold_size:]
    train_data_curr_fold = np.concatenate((train_data_curr_fold_head, train_data_curr_fold_tail))

    # epoch
    model = model.train()
    for curr_epoch in range(epoch):

        # train minibatch
        train_pred = []
        train_data_curr_fold = train_data_curr_fold[np.random.permutation(len(train_data_curr_fold))]
        for b in minibatch(train_data_curr_fold, batch_size): 
            train_batch_pred = train_batch(model, criterion, optimizer, b)
            train_pred.append(train_batch_pred)
        train_pred = np.concatenate(train_pred, axis=0)

        val_pred = []
        for b in minibatch(val_data_curr_fold, batch_size):
            val_batch_pred = val_batch(model, criterion, optimizer, b)
            val_pred.append(val_batch_pred)
        val_pred = np.concatenate(val_pred, axis=0)

        # calculate acc
        train_target = train_data_curr_fold[:, 1].reshape(-1)
        train_acc = cal_acc(train_pred, train_target)
        val_target = val_data_curr_fold[:, 1].reshape(-1)
        val_acc = cal_acc(val_pred, val_target)

        # print stats
        print(f"fold: {i}, epoch: {curr_epoch}, train acc: {train_acc}, val acc: {val_acc}")
    
    # test acc
    model = model.eval()
    test_pred = []
    for b in minibatch(test_data, batch_size):
        test_batch_pred = val_batch(model, criterion, optimizer, b)
        test_pred.append(test_batch_pred)
    test_pred = np.concatenate(test_pred, axis=0)
    test_target = test_data[:, 1].reshape(-1)
    test_acc = cal_acc(test_pred, test_target)
    test_f_score, test_percision, test_recall = cal_f(test_pred, test_target)
    print(f"fold: {i}, test acc: {test_acc}")
    print(f"fold: {i}, test percision: {test_percision}, test recall: {test_recall}, test f score: {test_f_score}")

