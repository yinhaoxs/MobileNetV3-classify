# coding=utf-8
# /usr/bin/env pythpn

'''
Author: yinhao
Email: yinhao_x@163.com
Wechat: xss_yinhao
Github: http://github.com/yinhaoxs

data: 2020-03-11 20:29
desc:
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict
from torchvision import transforms
from PIL import Image
from mobilenetv3 import mobilenetv3_large, mobilenetv3_small
import time
import os
from sklearn.metrics import confusion_matrix


def load_model(img_name):
    model = torch.load("small_best_model.pth")
    
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])])

    # evalute model
    model.eval()
    with torch.no_grad():
        img = Image.open(img_name).convert("RGB")
        img = transform(img)
        # add one dimension
        img = img.unsqueeze(0).cuda()
        out = model(img)
        pred = torch.max(out, 1)[1]

        return pred


if __name__ == "__main__":
    t = time.time()
    y_label, y_pred, k_label_list, nk_lable_list, k_pred_list, nk_pred_list  = list(), list(), list(), list(), list(), list()
    img_dir_k = "/data2/haoyin/datas/valid/k/"
    img_dir_nk = "/data2/haoyin/datas/valid/nk/"

    ## k字符集合
    for img_name in os.listdir(img_dir_k):
        img_path = os.path.join(img_dir_k+os.sep, img_name)
        k_label_list.append(0)
        pred = load_model(img_path)
        k_pred_list.append(pred.cpu().numpy())
        print("预测结果为:{}".format(pred.cpu().numpy()))
        # print("测试时间为:{}".format(time.time()-t))

    ## 非k字符集合
    for img_name in os.listdir(img_dir_nk):
        img_path = os.path.join(img_dir_nk+os.sep, img_name)
        k_label_list.append(1)
        pred = load_model(img_path)
        k_pred_list.append(pred.cpu().numpy())
        print("预测结果为:{}".format(pred.cpu().numpy()))
        
    y_label = k_label_list + nk_lable_list
    y_pred = k_pred_list + nk_pred_list
    matrix = confusion_matrix(y_label, y_pred)
    print("混淆矩阵为:{}".format(matrix))








