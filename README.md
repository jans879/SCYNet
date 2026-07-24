### SCYNet
Testing supersymmetric models at the LHC with neural networks. The neural network has 11 input parameters of the phenomenological minimal supersymmetric Standard Model (pMSSM-11). The network has one output value which is a chi^2. Lower chi^2s mean that the pMSSM-11 parameter point is compatible with measurements at the Large Hadron Collider (LHC). 

The network is the "direct approach" simple feed forward neural network from the SCYNet paper [1]. Detailes can be found in [1] or in my Master's thesis [2].

The neural networks have been hyper-parameter optimized. The code is mostly written in python3 with tensorflow.


File stucture

SCYNET_network. this folder includes the completely ready to use neural network. There are two sub-folders. 

 ->python. Call the network from a python code. 

 Example call: python3 get_chi2_13TeV_from_best_net.py M1  M2  M3  msq12 msq3 msl12 msl3 M_A A_0 mu tan(beta)

 where (M1  M2  M3  msq12 msq3 msl12 msl3 M_A A_0 mu tan(beta)) are the 11 parameters of the supersymmetric model. More details in [1,2]


 
 -> cpp. One can call the net from a cpp code. This option is very fast
SCYNet_train. This folder includes the code that was used to set up the network and train it.




[1] https://arxiv.org/abs/1703.01309
[2] Link to Master's thesis
