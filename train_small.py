# coding=utf-8
# /usr/bin/env pythpn

'''
Author: yinhao
Email: yinhao_x@163.com
Wechat: xss_yinhao
Github: http://github.com/yinhaoxs

data: 2020-03-12 22:31
desc:
'''

from __future__ import print_function, division

import os
import os

#多块使用逗号隔开
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from torchvision import models
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from mobilenetv3 import mobilenetv3_large, mobilenetv3_small

batch_size = 64
learning_rate = 0.00001
epoch = 100

train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dir = '/data2/haoyin/datas/train'
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=8,
                            pin_memory=True)

val_dir = '/data2/haoyin/datas/valid'
val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=8,
                            pin_memory=True)


# ---------------------模型保存----------------------------------
def save_checkpoint(model, path):
        torch.save(model, path)


# --------------------测试指标----------------------------------
# 测试计算指标，如混淆矩阵
def test(test_loader, model):
        # switch to evaluate mode
        model.eval()
        y_preds, y_trues = list(), list()
        for i, (images,labels ) in enumerate(test_loader):
                image_var = torch.autograd.Variable(images).cuda()
                # compute y_pred
                output = model(image_var)
                predicted = torch.max(output, 1)[1]
                print("预测的结果:{}, 原始的标签:{}".format(predicted, labels))
                print("预测:", type(predicted.cpu().numpy()))
                print("原始:", type(labels.numpy()))
                y_preds.append(predicted.cpu().numpy())
                y_trues.append(labels.numpy())
        matrix = confusion_matrix(list(y_trues), list(y_preds))
        print("混淆矩阵为:{}".format(matrix))
                

# --------------------训练过程---------------------------------
# 1.1 loading trained_model
pretrained_dict = torch.load("./pretrained/mobilenetv3-small-c7eb32fe.pth")
# 1.2 load parallel_mode
model = mobilenetv3_small(num_classes=2).cuda()
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# 1.3 优化器设置
optimizer = optim.Adam(model.classifier_x.parameters(), lr=learning_rate, weight_decay=1e-5)
loss_func = nn.CrossEntropyLoss()

Loss_list = []
Accuracy_list = []
best_acc = 0
for epoch in range(100):
        print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        for batch_x, batch_y in train_dataloader:
                batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
                out = model(batch_x)
                loss = loss_func(out, batch_y)
                train_loss += loss.data.item()
                pred = torch.max(out, 1)[1]
                train_correct = (pred == batch_y).sum()
                train_acc += train_correct.data.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
                train_datasets)), train_acc / (len(train_datasets))))

        # evaluation--------------------------------
        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        for batch_x, batch_y in val_dataloader:
                batch_x, batch_y = Variable(batch_x, volatile=True).cuda(), Variable(batch_y, volatile=True).cuda()
                out = model(batch_x)
                loss = loss_func(out, batch_y)
                eval_loss += loss.data.item()
                pred = torch.max(out, 1)[1]
                num_correct = (pred == batch_y).sum()
                eval_acc += num_correct.data.item()
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
                val_datasets)), eval_acc / (len(val_datasets))))

        Loss_list.append(eval_loss / (len(val_datasets)))
        Accuracy_list.append(100 * eval_acc / (len(val_datasets)))
       	 
        if eval_acc > best_acc:
                best_acc = eval_acc
                save_checkpoint(model, "small_best_model.pth")
                # test(val_dataloader, model)


