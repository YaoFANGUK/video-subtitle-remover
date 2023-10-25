# -*- coding: utf-8 -*-
# File   : test_numeric_batchnorm.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
# 
# This file is part of Synchronized-BatchNorm-PyTorch.

import unittest

import torch
import torch.nn as nn
from torch.autograd import Variable

from sync_batchnorm.unittest import TorchTestCase


def handy_var(a, unbias=True):
    n = a.size(0)
    asum = a.sum(dim=0)
    as_sum = (a ** 2).sum(dim=0)  # a square sum
    sumvar = as_sum - asum * asum / n
    if unbias:
        return sumvar / (n - 1)
    else:
        return sumvar / n


class NumericTestCase(TorchTestCase):
    def testNumericBatchNorm(self):
        a = torch.rand(16, 10)
        bn = nn.BatchNorm2d(10, momentum=1, eps=1e-5, affine=False)
        bn.train()

        a_var1 = Variable(a, requires_grad=True)
        b_var1 = bn(a_var1)
        loss1 = b_var1.sum()
        loss1.backward()

        a_var2 = Variable(a, requires_grad=True)
        a_mean2 = a_var2.mean(dim=0, keepdim=True)
        a_std2 = torch.sqrt(handy_var(a_var2, unbias=False).clamp(min=1e-5))
        # a_std2 = torch.sqrt(a_var2.var(dim=0, keepdim=True, unbiased=False) + 1e-5)
        b_var2 = (a_var2 - a_mean2) / a_std2
        loss2 = b_var2.sum()
        loss2.backward()

        self.assertTensorClose(bn.running_mean, a.mean(dim=0))
        self.assertTensorClose(bn.running_var, handy_var(a))
        self.assertTensorClose(a_var1.data, a_var2.data)
        self.assertTensorClose(b_var1.data, b_var2.data)
        self.assertTensorClose(a_var1.grad, a_var2.grad)


if __name__ == '__main__':
    unittest.main()
