# SCYNet

## Overview
The Large Hadron Collider (LHC) collides protons at record-high energies. By studying the outcomes of these collisions physicists gain insights into the smallest building blocks of nature and the fundamental forces governing their interactions.

One of the main goals of the LHC is to search for physics beyond the well-established Standard Model (SM) of particle physics. One of the leading frameworks for extending the SM is the Minimal Supersymmetric Standard Model (MSSM). Here, we consider a specific version of the MSSM, called the pMSSM-11, which introduces 11 new parameters in addition to those of the SM.

To test the pMSSM-11, we need to compare its predictions with measurements from the LHC. For a given point in the 11-dimensional pMSSM-11 parameter space, this is typically done by simulating the corresponding proton-proton collisions and comparing the predicted results with the LHC measurements. This comparison can be quantified using a χ² statistic, where lower (higher) χ² values indicate better (worse) agreement between the pMSSM-11 parameter point and the experimental data.

The problem is that conventional methods can take $\mathcal{O}(\mathrm{hours})$ to calculate the χ² value for a single pMSSM-11 parameter point. In a global fit, however, we need to evaluate potentially billions of parameter points in order to identify the region of the 11-dimensional parameter space that provides the best agreement with the LHC data. This makes conventional approaches computationally very expensive.

**The idea:** We train neural networks on a computationally feasible number of simulated pMSSM-11 parameter points. Once trained, the neural networks can perform the theory–experiment comparison on a vastly shorter timescale, calculating the χ² value in milliseconds rather than hours. This makes it possible to explore the pMSSM-11 parameter space much more efficiently. We dub the neural network **Susy Calculating Yield Net (SCYNet)**.


We provide two neural networks for two LHC collision energies, 8 TeV and 13 TeV. In the following, we summarize the methodology, network architecture, and training procedure for the 8 TeV network. The 13 TeV network is trained using a very similar strategy. Further details can be found in Refs. [1,2].


We show the network architecture in Fig. 1. The network is a fully connected feed forward neural network. The architecture shown in the figure was found through an extensive hyperparameter scan. More details can be found further below.

<p align="center">
  <img src="training_code/network_architecture.png" alt="Network architecture" width="800"><br>
  <em> Figure 1. Neural network architecture </em>
</p>

In Fig. 2, we show a histogram of all χ² values in the full dataset used to train and validate the model. The histogram contains a total of $\mathcal{O}(2\times10^5)$ entries. Each χ² value corresponds to a point in the 11-dimensional pMSSM-11 parameter space, and calculating the χ² value for each parameter point took several hours. We performed these simulations on a computing cluster.
Two distinct peaks appear around χ²$\approx 40$ and χ²$ \approx 100$. The points around χ²$ \approx 40$ are in good agreement with the LHC measurements, whereas the points around χ²$ \approx 100$ are strongly disfavored and in significant tension with the measurements.
The peaked structure arises from the way we sampled the 11-dimensional parameter space. In other words, the sampling procedure results in many more points with χ² values in these two regions than with values between the two peaks.

<p align="center">
  <img src="training_code/data_histogram.png" alt="Data distribution" width="600"><br>
  <em> Figure 2. Histogram of the target χ² distribution</em>
</p>

In Fig. 3, we show the mean error on the validation set after each training epoch. The solid black line represents the overall mean error, while the other lines show the mean error within different target ranges. The boundaries of these target ranges are indicated by the vertical dashed lines in the histogram in Fig. 2.
We observe that the mean error is generally larger in target ranges containing fewer data points. We refer to this as the **rare target learning problem (RTLP)**. This is a general feature that we have observed: the network learns target values more accurately when they occur more frequently in the training dataset.
We have explored several approaches to mitigate the RTLP (see below for more details). Although these approaches improve the performance in the less populated target ranges, we have not been able to fully eliminate the effect and achieve approximately equal mean errors across all target ranges.

<p align="center">
  <img src="training_code/mean_error_vs_epochs.png" alt="Mean error" width="700"><br>
  <em> Figure 3. Mean error with respect to the training epoch</em>
</p>


The provided code implements a simple, fully connected feed-forward neural network. We have implemented it using TensorFlow and Python 3. The network is highly flexible, allowing users to manually configure its architecture and training. The following properties can be adjusted:

- Number of hidden layers and neurons in each layer

- Activation functions in each layer (tanh, sigmoid, relu, etc.)

- Cost function (quadratic, cross-entropy, quadratic-clever (linear for small errors and quadratic for larger errors), etc.)

- Minimization algrotihm for cost function (Gradient Descent, Adam optimizer, etc.)

- Batch size for mini-batch learning

- Methods in order to avoid overfitting (L2 regularization (adds $\lambda/(2N_{\rm train})\sum {weights}$ to cost function), dropout, etc.)

- Weight and bias initialization: We initialize the weights which connect layer $l$ and $l-1$ with a gaussian distribution which has mean zero and standard deviation $1/N_{l-1}$, where $N_{l-1}$ are the number of neurons in layer $l-1$. The biases are initialized with a standard normal distribution. Other initialization procedures can easily be implemented if needed.

- Feature scaling: It can be beneficial for the training process to apply a transformation to the input $x_i, i=1\cdot 11$ and output variables $y=$χ². The code is written such that the implemented transofromations can easily be adjusted. The transformation applied to the outputs must be invertible in order to be able to transform the neural network's output back to the original scale. The transformation of the inputs does not necessarily have to be invertible. When using a tanh activation function in the output neuron we use a so-called modified Z-score transformation:

$$
\hat{y}= \left(y-\mu\right)/\sigma
$$

and then
 
$$
\hat{\hat{y}} = \hat{y}/{\rm max} |\hat{y}|
$$

where $\mu = (y_{\rm min}+y_{\rm max})/2$ and $\sigma$ is the standard deviation. We use this particular expression for $\mu$ such that $y_{\rm min}$ corresponds to $-1$ and $y_{\rm max}$ to $+1$ thereby covering the entire thanh range. For the input values we use the same transormation but in this case $\mu$ represents the mean of all input values. This corresponds to the standard Z-score normalization.

- Learning slowdown: After every 10 training epochs the code checks wether the slope of a line which fitted to the last 10 mean validation errors exceeds a specific threshold. If this is the case the learning rate of the minimization algorithm is reduced by a facotr of 2.

- Exponential damping: We multiply each term in the cost function with an exponential term that gives more weight to small targets. For example for the quadratic cost function: 

$$
\frac{1}{N_{\rm train}}\sum_{i=1,\cdots N_{\rm train}}(y_i-o_i)^2 \, e^{-5\frac{y_i}{y_{\rm max}}}
$$

where $o_i$ the output of the neural network and $y_i$ is the desired target value.



We ran **hyperparameter scans** to identify the optimal network structure and training procedure. In the following we present a table showing all hyperparameters summarizing all hyperparametersthat were optimized for the 8 TeV energy neural network. In the last column we show the optimal values that were found. 
For each hyperparameter combination in the table we trained a seperate neural network. In total we tested 41472 hyperparameter configurations, corresponding to 41472 independently trained neural networks.
To avoid overfitting to the hyperparameter scan we use a different training and validation set for each network. This ensures that the hyperparameter optimization is not biased toward a particular validation set.


|Hyperparameter | Scanned | Best  |
|-----------|-------------|---------|
|Number of hidden layers | 2, 3, 4, 5 | **4** |
|Number of neurons in hidden layers | 50, 150, 450 | **150** |
|Cost function | quadratic, cross | **quadratic** |
|Exponential damping | on, off | No preference
|Batch size | 80, 500, 3000 | **500** |
|lambda | $10^{-3},10^{-4},10^{-5},10^{-6}$ | **$10^{-5}$** |
|Learning rate | $10^{-1},10^{-2},10^{-3},10^{-4}$ | **$10^{-3}$** |
|Dropout probablitty 1 | 0.9, 0.95, 1 | **1** |
|Dropout probablitty 2 | 0.9, 0.95, 1 | **1** |
|Activation in last layer | tanh, linear | **tanh** |


The activation functions in the hidden layers are all tanh. We used the Adam optimizer and its hyperparameters (except the learning rate) are set to their default values. The two dropout probabilities are applied alternating to the hidden layers.


Even with the optimized hyperparameters we still encounter the RTLP. We have therefore further tried to mitigate the RTLP in several ways:


- Artificial data set extension: One can artificially duplicate the pMSSM-11 parameter points that yield χ² values in a rare target area. The duplicated points are assigned the same χ² value as their original points, but one component of the 11-dimensional parameter point is slightly modified.

- Sequence learning: The neural network is not always trained with the full training set. For example, the network may be trained on the full training set for two epochs, followed by a training epoch whith only training data from the rare target regions. Training only with the rare target data is done using a reduced learning rate.

- Additional sampling in rare target areas: One can sample new points in the 11 dimensional parameter space that lead to target values (χ²) in rare target areas. A closely related approach is to sample new points around existing parameter points that yield χ² values in the rare target regions.


The first two options above are included in the SCYNet code and can be activated by `setting sequence_learning = "True"` and `extend_data_artificially = "True"`. The additionally sampled data is per default included in the provided data set.

We carried out a hyperparameter scan specifically aimed at mitigating the RTLP. The hyperparameters from the previous scan were fixed to the best-performing values identified in that scan. The following table shows the hyperparameters that were varied in the scan to mitigate the RTLP.

|Hyperparameter | Scanned | Best  |
|-----------|-------------|---------|
|Additional sampling | yes, no | **yes** |
|Artificial Extension | yes, no | **no** |
|Sequence learning | yes, no | **yes** |
|Multiply number of neurons in first hiden layer by | 2, 1.5, 1 | **2** |
|Multiply number of neurons in other hiden layer by | 2, 1.5, 1 | **2** |
|Multiply batch size by | 1.5, 1 | **1.5** |



Additional sampling and sequence learning help to improve the performance of the network in the rare target ranges. However, the artificially extending of the data set -- as implemented in the code -- did not lead to an improvement in the RTLP. Increasing the number of neurons in the hidden layers and the batch size was also found to be beneficial which is reasonable given that more training data  are available.

We want to point out that our studies on how to improve the RTLP were preliminary and there are likely many other approaches that could be explored. It is important to say that wile some methods helped mitigate the RTLP, none of them fully resolved the problem. The mean error in the rare taregt ranges remained larger than the mean error in the other ranges. I plan to investigate this behavior more in the future and explore wether new network architectures can further improve the RTLP.

For example, it would be interesting to explore if more modern transformer-based architectures can significantly improve the RTLP. The models that we have used here are still relativley small and have $\mathcal{O}(10^5)$ parameters. It would therefore be interesting to study wether models with many more parameters can achieve better performance in the rare target regions and further mitigate the RTLP.
























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
