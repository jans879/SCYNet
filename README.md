# SCYNet
Testing supersymmetric models at the LHC with neural networks

This repository includes the "direct approach" simple feed forward neural network from the SCYNet paper. Detailes can be found in [1] or in my Master's thesis [2]

The neural networks have been hyper-parameter optimized. 


File stucture

SCYNET_network. this folder includes the completely ready to use neural network. There are two sub-folders. 
 -> cpp. One can call the net from a cpp code. This option is very fast
 ->python. Call the network from a python code. Not optimal for global fitting because the net has to be loaded many times during the global fit.

SCYNet_train. This folder includes the code that was used to set up the network and train it.




[1] https://arxiv.org/abs/1703.01309
[2] Link to Master's thesis
