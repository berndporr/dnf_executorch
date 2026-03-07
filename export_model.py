# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright (c) Bernd Porr
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


# DNF encoder
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(100, 1)

    def forward(self, x):
        return self.linear1(x)

class GradientLoss(nn.Module):
    # The derivative of the loss with respect to the output
    # is simply the gradient!
    def forward(self, remover, inputSignal):
        error = inputSignal - remover;
        return (remover * error).sum()

class TrainingNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.criterion = GradientLoss()

    def forward(self, input, grad):
        pred = self.net(input)
        loss = self.criterion(pred,grad)
        return loss, pred.detach()


def _export_model():
    net = TrainingNet(Net())
    x = torch.randn(1, 100)
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
    external_mutable_weights = False
    ep = to_edge(ep)
    ep = ep.to_executorch(
        config=ExecutorchBackendConfig(
            external_mutable_weights=external_mutable_weights
        )
    )
    return ep

def main() -> None:
    pte_filename = "dnf.pte"
    torch.manual_seed(0)
    program = _export_model()

    with open(pte_filename, "wb") as fp:
        program.write_to_file(fp)

    from executorch.runtime import Runtime
    runtime = Runtime.get()
    method = runtime.load_program(pte_filename).load_method("forward")
    x = torch.randn(1, 100)
    outputs = method.execute([x, torch.ones(1)])
    print(outputs)


if __name__ == "__main__":
    main()
