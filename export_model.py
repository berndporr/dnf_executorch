# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright (c) Bernd Porr
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import argparse

import os

import torch
from executorch.exir import ExecutorchBackendConfig, to_edge

from executorch.extension.training.examples.XOR.model import Net, TrainingNet
from torch.export import export
from torch.export.experimental import _export_forward_backward


def _export_model():
    net = TrainingNet(Net())
    x = torch.randn(2, 100)

    # Captures the forward graph. The graph will look similar to the model definition now.
    ep = export(net, (x, torch.ones(1, dtype=torch.int64)), strict=True)
    # Captures the backward graph. The exported_program now contains the joint forward and backward graph.
    ep = _export_forward_backward(ep)
    # Lower the graph to edge dialect.
    ep = to_edge(ep)
    # Lower the graph to executorch.
    ep = ep.to_executorch(
        config=ExecutorchBackendConfig()
    )
    return ep


def main() -> None:
    torch.manual_seed(0)
    ep = _export_model()

    with open("dnf.pte", "wb") as fp:
        ep.write_to_file(fp)

if __name__ == "__main__":
    main()
