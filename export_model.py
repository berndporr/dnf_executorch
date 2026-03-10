# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright (c) 2026 Bernd Porr
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from executorch.exir import ExecutorchBackendConfig, to_edge
from executorch.extension.training.examples.XOR.model import Net, TrainingNet
from torch.export import export
from torch.export.experimental import _export_forward_backward
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# Noise ref delay line length
nTaps = 100

# Number of layers. 1=FIR/LMS filter.
nLayers = 3

# Nonlinearity
nonlin = nn.Tanh()

# The PTE filename:
pte_filename = "dnf_executorch.pte"


# DNF encoder
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq = nn.Sequential()
        b = np.exp(np.log(nTaps)/nLayers);
        nInput = nTaps
        for i in range(nLayers):
            if (i == (nLayers-1)):
                nOutput = 1
            else:
                nOutput = int(np.ceil(nTaps / np.pow(b,i+1)));
            print("Created layer",i,"with",nInput,"->",nOutput)
            l = nn.Linear(nInput, nOutput)
            nn.init.xavier_uniform(l.weight)
            nn.init.zeros_(l.bias)
            self.seq.add_module("Layer"+str(i),l)
            self.seq.add_module("Nonlin"+str(i),nonlin)
            nInput = nOutput

    def forward(self, x):
        return self.seq(x)

class TrainingNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.criterion = nn.MSELoss()

    def forward(self, noiseRef, noisySignal):
        remover = self.net(noiseRef)
        loss = self.criterion(noisySignal,remover)
        return loss, remover.detach()


def _export_model():
    net = TrainingNet(Net())
    x = torch.randn(1, nTaps)

    # Captures the forward graph. The graph will look similar to the model definition now.
    ep = export(net, (x, torch.ones(1)), strict=True)
    print("Forward graph:")
    print(ep.graph)
    print()

    # Captures the backward graph. The exported_program now contains the joint forward and backward graph.
    ep = _export_forward_backward(ep)
    print("Forward / backward graph:")
    print(ep.graph)
    print()

    # Lower the graph to executorch.
    ep = to_edge(ep)
    ep = ep.to_executorch()
    return ep

def main() -> None:
    torch.manual_seed(0)
    program = _export_model()

    with open(pte_filename, "wb") as fp:
        program.write_to_file(fp)

    from executorch.runtime import Runtime
    runtime = Runtime.get()
    method = runtime.load_program(pte_filename).load_method("forward")
    x = torch.randn(1, nTaps)
    outputs = method.execute([x, torch.ones(1)])
    print(outputs)


if __name__ == "__main__":
    main()
