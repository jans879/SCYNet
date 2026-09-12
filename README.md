# SCYNet

## Overview
This repository contains neural networks for testing supersymmetric models against measurements from the Large Hadron Collider (LHC). The networks take as input the 11 parameters of the phenomenological Minimal Supersymmetric Standard Model (pMSSM-11) and predict a single output value: a χ² statistic, see Fig. below. Lower χ² values indicate better agreement between a given pMSSM-11 parameter point and measurements from the LHC. 
We provide two networks for two LHC collision energies (8TeV and 13 TeV). In the following we summarize the details on the methodology, network architecture, and training procedure. For more detils see Refs. [1,2].

<p align="center">
  <img src="training_code/network_architecture.png" alt="Network architecture" width="800"><br>
  <em>Histogram of the target χ² distribution</em>
</p>

In the following figure, we show a histogram of all χ² values in the full dataset used to train and validate the model. Because of the way the data was generated in the 11-dimensional parameter space, there are two clear peaks around 40 and 100. In other words, there are many more 11-dimensional data points with lead to target χ² values around these two regions than with target values in between the peaks. 

<p align="center">
  <img src="training_code/data_histogram.png" alt="Data distribution" width="600"><br>
  <em>Histogram of the target χ² distribution</em>
</p>

In the following figure, we show the mean error on the points in the validation set after each training epoch. The solid black line shows the overall mean error, while the other lines show the mean error in different target ranges. The different target ranges shown are marked by vertical dashed lines in the histogram above.
We observe that the mean error is generally larger in the target ranges that contain less data points. We call this the **rare target learning problem (RTLP)**. It is a general feature that we have observed: the network learns targets better when they appear more frequently in the dataset.
We have tried to mitigate the RTLP in several ways (more details below). While we were able to improve the RTLP, we were not able to fully resolve it such that the mean error in all target ranges is roughly the same.

<p align="center">
  <img src="training_code/mean_error_vs_epochs.png" alt="Mean error" width="700"><br>
  <em>Mean error with respect to the training epoch</em>
</p>


The provided code is simple fully connected feed forward neural network. We have implemented it with Tensorflow and python3. The network is very flexible and the user can manually configure the neural network. The following properties can easily be adjusted:

- Number of hidden layers and neurons in each layer

- Activation functions in each layer (tanh, sigmoid, relu, etc.)

- Cost function (quadratic, cross-entropy, quadratic-clever(linear for small errors and quadratic for larger errors), etc.)

- Minimization function for cost function (Gradient Descent, Adam optimizer, etc.)

- Batch size for mini-batch learning

- Methods in order to avoid overfitting (L2 regularization (adds $\lambda/(2N_{\rm train})\sum $ to cost function), dropout, etc.)

- Weight and bias initialization: We initialize the weights which connect layer l and l-1 with a gaussian distribution which has mean zero and standard deviation $1/N_{l-1}$, where $N_{l-1}$ are the number of neurons in layer l-1. The biases are initialized with a standard normal distribution. Other initialization procedures can easily be implemented if needed.

- Feature scaling: It can be beneficial for the trainign if we apply a transformation to the input $x_i, i=1\cdot 11$ and output values $y=\chi^2$. The code is written in a way that it easy to adjust the transofromation. The transformation which one applies on the outputs has to be invertible in order to be able to back transform the outputted values of the neural net. The transformation on the inputs does not have to be invertible. When using a tanh activation function in the output neuron we use a so-called modified Z-score transformation. 
$$
\hat{y}= \left(y-\mu\right)/\sigma
$$
and then 
$$
\hat{\hat{y}} = \hat{y}/{\rm max} |\hat{y}|
$$
where $\mu = (y_{\rm min}+y_{\rm max})/2$ and $\sigma$ is the standard deviation. We use this expression for $\mu$ so that $y_{\rm min}$ corresponds to $-1$ and $y_{\rm max}$ to $+1$ and the entire target range of tanh is covered. For the input values we use the same transormation but this time $\mu$ really represents the mean of all input values (this is the normal Z-score normalization).

- Learning slowdown: After 10 learning epochs we check if the slope of a line which has been fitted to the last 10 validation errors is larger than some threshold. If this is the case the learning rate of the minimization algorithm will be reduced by 1/2.

- Exponential damping: We multiply each squared term in the cost function with an exponential term that gives more weight to small targets. For example for the quadratic cost function:
$$
\frac{1}{N_{\rm train}}\sum_{i=1,\cdots N_{\rm train}}(y_i-o_i)^2 \, e^{-5\frac{y_i}{y_{\rm max}}}
$$
where $o_i$ the output of the neural network and $y_i$ is the desired target value.



We ran sophisticated **hyperparameter scans** to identify the optimal network structure and training procedure. In the following we present a table showing all hyperparameters that have been optimized for the 8 TeV energy neural network. We also show the optimal hyperparameter that were found. We have trained a neural network for each combination of values shown in the table. In total we have tested 77760 hyperparameter configurations, i.e., we trained 77760 differen neural networks. To avoid hyperparameter overfitting we use a different training and validation set for each network that we have trained during the hyperparameter scan.


|Hyperparameter | Scanned | Best  |
|-----------|-------------|---------|
|Number of hidden layers | 2,3,4,5 | **4** |
|Number of neurons in hidden layers | 50,150,450 | **150** |
|Cost function | quadratic, cross | **quadratic** |
|Exponential damping | on, off | No preference
|Batch size | 80,500,3000 | **500** |
|lambda | 10^{-3},10^{-4},10^{-5},10^{-6} | **10^{-5}** |
|Learning rate | 10^{-1},10^{-2},10^{-3},10^{-4} | **10^{-3}** |
|Dropout probablitty 1 | 0.9,0.95,1 | **1** |
|Dropout probablitty 2 | 0.9,0.95,1 | **1** |
|Activation in last layer | (tanh, linear) | **tanh** |


The activation functions in the hidden layers are all tanh. The other Adam optimizer hyperparameters (except the learning rate) are set to their default values. The two dropout probabilities are applied alternating to the hidden layers.


We have tried to mitigate the RTLP in several ways:


- Artificial extension: One duplicates the pMSSM-11 parameter points which lead to χ² values in a rare target area. The duplicated points get the same χ² as their original points, but one component of the 11-dimensional parameter point is slightly modified 

- Sequence learning: One trains the neural net not always with the full training set. For example for two epochs one trains the neural net with the full training set and then for one epoch one uses only the data in the training set which has target values in the rare areas. This will be repeated over and over. The training with the rare target data happens with a reduced learning rate.

- Additional sampling in rare target areas: One can identify areas in the 11 dimensional parameter space which lead to target values (χ²) which lie in rare target areas. Then one can sample especially new points in these areas. Another very similar approach is to sample around existing parameter points which lead to χ² values in the rare target areas.


The first two options above are included in the SCYNEt code and can be activated easily by setting sequence_learning = "True" and extend_data_artificially = "True". The additionally sampled data is per default included in the provided data set.

We carried out a hyperparameter scan specifically to mitigate the RTLP. The hyperparameters of the previous scan are set to the best case parameters that were found. The following table shows the hyperparameters that were scanned in order to avoid the RTLP

|Hyperparameter | Scanned | Best  |
|-----------|-------------|---------|
|Additional sampling | yes, no | **yes** |
|Artificial Extension | yes, no | **no** |
|Sequence learning | yes, no | **yes** |
|Multiply number of neurons in first hiden layer by | 2, 1.5, 1 | **2** |
|Multiply number of neurons in other hiden layer by | 2, 1.5, 1 | **2** |
|Multiply batch size by | 1.5, 1 | **1.5** |



Additional sampling and Sequence learning helps to imporve the performance of the network in those ranges. However the artificial extension of the data -- at least as we have implemented it -- did not help to improve the RTLP. Increasing the number of neurons in the hidden layers and the batch size was also found to be beneficial which makes sense when we have more data availabel.
We want to point out that our studies were just preliminary and we could have probably played around with it much more. However, it is important to say that wile some methods helped to mitigate the RTLP, non of them fully resolved it. The mean error in the rare taregt ranges was still larger than the mean error in the other ranges. I plan to investigate this behaviour more in the future and try if new network architectures can improve the RTLP.
It would for example be interesting to explore if more modern transformer architectures can significantly improve the RTLP. The models that we have used here are still relativley small and have $\mathcal{O}(10^5)$ parameters. It would be interesting to see if models with many more parameters can improve the RTLP.


All steps that we have described here are for the 8 TeV network. We have done similar steps for the 13 TeV network that are described in Refs. [1,2].






















## Repository Structure

```text
SCYNet/
├── README.md
├── training_code/
│   ├── data/
│       └── unpack_data.tar.gz
│   ├── network/
│       ├── network_performance_data/
│       ├── network_performance_plots/
│       ├── plot_training.py
│       ├── read_in_Data.py
│       └── train_SCYNET.py
│   ├── data_histogram.png
│   ├── mean_error_vs_epochs.png
│   └── network_architecture.png
├── trained_networks/
    ├── cpp
│   └── python
        ├── get_chi2_13TeV_from_best_net.py
        ├── net_13TeV.ckpt
        └── transformations.py
└── docs/
    └── thesis.pdf
```


## training_code

In order to train the network from scratch first unpack the data in /data and put it in the folder /data.

Then run the network with 
```text
python3 train_SCYNET.py
```
After the network has finished training it produces output data files in /network/network_performance_data/.
You can visualize the network performance with
```text
plot_training.py
```
Which produces network performance plots in /network/network_performance_plots/



## trained_networks (python)

This folder contains the trained networks that are ready to use. This i for a 13 TeV network, i.e. the network compares the pMSSM-11 model to measurements at the LHC with 13 TeV center of mass energy

Example call: python3 get_chi2_13TeV_from_best_net.py M1  M2  M3  msq12 msq3 msl12 msl3 M_A A_0 mu tan(beta)

where (M1  M2  M3  msq12 msq3 msl12 msl3 M_A A_0 mu tan(beta)) are the 11 parameters of the supersymmetric model. More details in [1,2]


## trained_networks (cpp)

We also provide a framework that allows the network to be embedded in C++ code and called directly from the C++ implementation. This can be useful in applications where speed is important. For example, in global fits, where one aims to identify the best-fit parameters of the pMSSM-11, one typically needs to scan over a large parameter space.

```text
make
make run
./run
```










[1] https://arxiv.org/abs/1703.01309

[2] See docs folder
