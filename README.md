# Deep Neuronal Filter (DNF) -- executorch version

![alt tag](dnf_diagram.png)

The [deep neuronal filter (DNF)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0277974) 
is an extension of the classical LMS FIR adaptive noise canceller.
Instead of using an adaptive FIR filter the DNF uses a realtime adaptive deep neural network.
This requires high performance sample by sample forward and backward processing. The new
(still experimental) feature of executorch is ideal for this purpose because it has realtime
backprop doing gradient descent with a loss function.

## Prerequisites

Install [executorch](https://github.com/pytorch/executorch) as a library on your system:

```
git clone https://github.com/pytorch/executorch
cd executorch
source ~/venv/bin/activate
python install_executorch.py
cmake --preset linux -DEXECUTORCH_BUILD_EXTENSION_TRAINING=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DEXECUTORCH_ENABLE_LOGGING=ON -DEXECUTORCH_LOG_LEVEL=ERROR .
cd cmake-out
make
sudo make install
```

Note for now logging should be enabled as the training extension is (still) experimental and
errors should be reported back to executorch.

## How to compile

Type:

```
cmake .
make
```
to compile the library and the demos.

## Unit tests

```
make test
```

## How to install

```
sudo make install
```

## How to use it

### Create the DNF model

Create the pte file with the `export2executorch` python script:

```
import torch
import export2executorch
nTaps = 50
nLayers = 5
export2executorch.dnf2executorch("dnf_executorch.pte",nTaps,nLayers)
```
where `nTaps` is the number of taps of the delay line feeding into the
deep net with `nLayers` layers.

### Init

```
DNF_executorch dnf(`export2executorch`, mu);
dnf.setLearning(true);
```
where `mu` is the learning rate (typically around 0.01). You cannot change the
learning rate later but you can switch learning on and off with `dnf.setLearning(bool);`.

### Realtime noise cancellation sample by sample

This code snipplet should happen, for example, in your sample-by-sample callback:
```
const double output_signal = dnf.filter(noisy_input_signal, ref_noise);
```
where `ref_noise` is the noise you'd like to be removed from `noisy_input_signal`.

## Example

[Simple instructional example which removes 50Hz from an ECG](ecg_filt_demo) with just
one layer which is identical to an FIR LMS filter.

See also the tests which do learning with one layer and five layers.

## Class documentation

The doxygen generated files are here: https://berndporr.github.io/dnf_executorch/

## Credits

- Bernd Porr who has written this version of the DNF
- Sama Daryanavard wrote the original implementation of the DNF which is here: https://github.com/berndporr/deepNeuronalFilter
