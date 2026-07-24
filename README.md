# SCYNet

## Overview
This repository contains the neural network for testing supersymmetric models against measurements from the Large Hadron Collider (LHC). The networks take as input the 11 parameters of the phenomenological Minimal Supersymmetric Standard Model (pMSSM-11) and predict a single output value: a χ² statistic. Lower χ² values indicate better agreement between a given pMSSM-11 parameter point measurements from the LHC.

The implemented model corresponds to the direct approach described in the SCYNet paper [1]. Additional details on the methodology, network architecture, and training procedure can be found in Ref. [1] and in my Master's thesis [2].

The neural network was optimized through extensive hyperparameter scans to achieve the best possible performance. The codebase is written primarily in Python 3 and uses TensorFlow for training and inference.

## Repository Structure

```text
SCYNet/
├── README.md
├── training_code/
│   ├── TOUPLOAD.py
│   └── transformations.py
├── trained_networks/
    ├── cpp
│   └── python
        ├── get_chi2_13TeV_from_best_net.py
        ├── net_13TeV.ckpt
        └── transformations.py
└── docs/
    └── thesis.pdf
```

## trained_networks (python)

This folder contains the trained networks that are ready to use. This i for a 13 TeV network, i.e. the network compares the pMSSM-11 model to measurements at the LHC with 13 TeV center of mass energy

Example call: python3 get_chi2_13TeV_from_best_net.py M1  M2  M3  msq12 msq3 msl12 msl3 M_A A_0 mu tan(beta)

where (M1  M2  M3  msq12 msq3 msl12 msl3 M_A A_0 mu tan(beta)) are the 11 parameters of the supersymmetric model. More details in [1,2]


## trained_networks (cpp)

We also provide a framework that allows the network to be embedded in C++ code and called directly from the C++ implementation. This can be useful in applications where speed is important. For example, in global fits, where one aims to identify the best-fit parameters of the pMSSM-11, one typically needs to scan over a large parameter space.

```text
run
make run
./run
```


## training_code

The training code is not ready to use on any computer yet. I still need to upload the data that is necessary to train the network








[1] https://arxiv.org/abs/1703.01309

[2] Link to Master's thesis
