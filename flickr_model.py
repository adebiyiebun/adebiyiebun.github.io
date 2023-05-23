#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from skimage import transform
import numpy as np


from PIL import Image
import time
import math
import copy

import torch.optim as optim
from torch.autograd import Variable
from torchvision import models

import warnings
warnings.filterwarnings("ignore")
import random
from scipy.stats import spearmanr



class APPModel(nn.Module):
    def __init__(self, keep_probability, inputsize):

        super(APPModel, self).__init__()

        self.fc1_1 = nn.Linear(inputsize, 2048)
        self.bn1_1 = nn.BatchNorm1d(2048)
        self.drop_prob = (1 - keep_probability)
        self.relu1_1 = nn.PReLU()
        self.drop1_1 = nn.Dropout(self.drop_prob)
        self.fc2_1 = nn.Linear(2048, 1024)
        self.bn2_1 = nn.BatchNorm1d(1024)
        self.relu2_1 = nn.PReLU()
        self.drop2_1 = nn.Dropout(p=self.drop_prob)
        self.fc3_1 = nn.Linear(1024, 5)
        self.bn3_1 = nn.BatchNorm1d(5)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """
        out_p = self.fc1_1(x)
        out_p = self.bn1_1(out_p)
        out_p = self.relu1_1(out_p)
        out_p = self.drop1_1(out_p)
        out_p = self.fc2_1(out_p)
        out_p = self.bn2_1(out_p)
        out_p = self.relu2_1(out_p)
        out_p = self.drop2_1(out_p)
        out_p = self.fc3_1(out_p)
        out_p = self.bn3_1(out_p)
        out_p = self.tanh(out_p)

        return out_p


class APRModel(nn.Module):
    def __init__(self, keep_probability, inputsize):

        super(APRModel, self).__init__()

        self.fc1_1 = nn.Linear(inputsize, 2048)
        self.bn1_1 = nn.BatchNorm1d(2048)
        self.drop_prob = (1 - keep_probability)
        self.relu1_1 = nn.PReLU()
        self.drop1_1 = nn.Dropout(self.drop_prob)
        self.fc2_1 = nn.Linear(2048, 1024)
        self.bn2_1 = nn.BatchNorm1d(1024)
        self.relu2_1 = nn.PReLU()
        self.drop2_1 = nn.Dropout(p=self.drop_prob)
        self.fc3_1 = nn.Linear(1024, 5)
        self.bn3_1 = nn.BatchNorm1d(5)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """
        out_p = self.fc1_1(x)
        out_p = self.bn1_1(out_p)
        out_p = self.relu1_1(out_p)
        out_p = self.drop1_1(out_p)
        out_p = self.fc2_1(out_p)
        out_p = self.bn2_1(out_p)
        out_p = self.relu2_1(out_p)
        out_p = self.drop2_1(out_p)
        out_p = self.fc3_1(out_p)
        out_p = self.bn3_1(out_p)
        out_p = self.tanh(out_p)

        return out_p


class convNet(nn.Module):
    def __init__(self,apr,app):
        super(convNet, self).__init__()
        self.APR=apr
        self.APP=app
    def forward(self,x):
        # x1 is apr and x2 is app
        x1=self.APR(x)
        x2=self.APP(x)
        return x1, x2


