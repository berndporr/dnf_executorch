# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright (c) Bernd Porr
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.nn import functional as F

# DNF encoder
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(100, 10)
        self.linear2 = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear2(self.linear1(x))


# On device training requires the loss to be embedded in the model (and be the first output).
# We wrap the original model here and add the loss calculation. This will be the model we export.
class TrainingNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, input, loss):
        pred = self.net(input)
        return loss, pred.detach().argmax(dim=1)
