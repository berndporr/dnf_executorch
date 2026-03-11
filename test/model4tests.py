# Copyright (c) 2026 Bernd Porr

import sys
import torch
sys.path.append('..')
import dnf2executorch
from executorch.runtime import Runtime

def main() -> None:
    pte_filename = "dnf_executorch.pte" 
    nTaps = 50
    nLayers = 1
    dnf2executorch.dnf2executorch(pte_filename,nTaps,nLayers)

    runtime = Runtime.get()
    method = runtime.load_program(pte_filename).load_method("forward")
    x = torch.randn(1, nTaps)
    outputs = method.execute([x, torch.ones(1)])
    print(outputs)


if __name__ == "__main__":
    main()
