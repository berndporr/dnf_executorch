# Copyright (c) 2026 Bernd Porr

import sys
import torch
sys.path.append('..')
import export2executorch

def main() -> None:
    pte_filename = "dnf_executorch.pte" 
    nTaps = 50
    nLayers = 1
    export2executorch.dnf2executorch(pte_filename,nTaps,nLayers)

if __name__ == "__main__":
    main()
