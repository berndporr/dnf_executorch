# Copyright (c) 2026 Bernd Porr
# BSD license

import torch
from executorch.exir import to_edge
from torch.export import export
from torch.export.experimental import _export_forward_backward
import torch.nn as nn
import numpy as np

# DNF encoder
class Net(nn.Module):
    def __init__(self,nTaps,nLayers,nonlin):
        super().__init__()

        torch.manual_seed(42)
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

def dnf2executorch(pte_filename, nTaps = 50, nLayers = 1, nonlin = nn.Tanh()):
    """
    Creates the pte file of the DNF.
    pte_filename: the filename of the pte file to be exported.
    nTaps: Noise ref delay line length.
    nLayers: Number of layers. 1=FIR/LMS filter.
    nonlin: Nonlinearity.
    """

    net = TrainingNet(Net(nTaps,nLayers,nonlin))
    x = torch.randn(1, nTaps)

    # Captures the forward graph.
    ep = export(net, (x, torch.ones(1)), strict=True)
    print("Forward graph:")
    print(ep.graph)
    print()

    # Captures the backward graph, too.
    ep = _export_forward_backward(ep)
    print("Forward / backward graph:")
    print(ep.graph)
    print()

    # Lower the graph to executorch.
    ep = to_edge(ep)
    ep = ep.to_executorch()

    with open(pte_filename, "wb") as fp:
        ep.write_to_file(fp)


def main() -> None:
    torch.manual_seed(0)
    pte_filename = "dnf_executorch.pte" 
    nTaps = 50
    dnf2executorch(pte_filename,nTaps)
    from executorch.runtime import Runtime
    runtime = Runtime.get()
    method = runtime.load_program(pte_filename).load_method("forward")
    x = torch.randn(1, nTaps)
    outputs = method.execute([x, torch.ones(1)])
    print(outputs)


if __name__ == "__main__":
    main()
