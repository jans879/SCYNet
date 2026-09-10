#################################
# This code was written by Jan Schuette-Engel (janschue@berkeley.edu).
# This is updated code of the SCYNET project from 2017. 
# I polished the code and update it. But I do not substentially change it 
# such that it is still documented by my Master's thesis and the SCYNET paper.
# Please cite arxiv:1703.01309 if you want to use this code
#################################

#####################
# imports
#####################

import numpy as np
from random import randint
import random
import math
import sys
import scipy.interpolate as ip_scipy
import os
import os.path
from array import array

# import tensorflow.v1 because the whole project was done with tensorflow v1
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#import plot_functions as pl
import read_in_Data as rd
import plot_training as pt


###############
#Read in data #
###############

energy = "13" # can chose 8 or 13 


file_dir="./../data"
file_name="" #this will be set below


N_in = 11 #input nodes, the network has 11 inputs (the parameters of the supersymmetric model)
N_out = 1 # output nodes (one output, the chi squared)


# filename for the training and validation information. 
# each line in this file represents one training point and each line must have the form
# array_id x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 y1
# where we used the example of a net with 11 inputs and one output 
file_name = energy+"TeV_chi2_disjoint"

train_ys_only_in_range = [0,100] #restrict to a specific target range

#variables for a smooth cut
cut_output_max = 100.0 # smooth cut (maximal value of the targets) (here in the example cut_output_max coincides with train_ys_only_in_range[1], but this must not be the case.
smooth_cut_range = 25.0 # cut range for the smooth cut. If set to -1 the cut will be hard. Do not set to zero for a hard cut. Set it -1 for a hard cut # 40, 25


cut_output_min = 11.9 # hard cut at cut_output_min, furthermore the backtransformed values of the network are never smaller than cut_output_min

#get informations like the mean errors also for the specific ranges which are specified below
y_ranges = None
if energy == "8":
    y_ranges = [[0.0,38.0],[38.0,42.0],[42.0,70.0],[70.0,95.0],[95.0,100.0001]]
elif energy == "13":
    y_ranges = [[0.0,53.5],[53.5,56.0],[56.0,70.0],[70.0,95.0],[95.0,100.0001]]

extend_data_artificially = "False"

extend_data_artificially_in_ranges = [1]*len(y_ranges) #no artificial extension per default
if extend_data_artificially == "True":
    extend_data_artificially_in_ranges = [3,1,1,3,1] #extend data in first and fourth range by a factor of 3

# if one doesn't want to consider some array ids's for some reasons
do_not_consider_these_array_ids= [[-1]] # per default consider all array id's
do_not_consider_these_array_ids = [[125000,1100000000]] # here one would not consider points with array ids between 125000 and 1100000000


sequence_learning = "False"

outputfolder = "./test_output" # specify the folder where everything should be placed

if not os.path.isdir(outputfolder):
    print("make output folder")
    os.makedirs(outputfolder)

input_net = "" #if "" no input net will be taken otherwise one has to specify the path to the input net, e.g. "./net.ckpt"
output_net = outputfolder+"/net.ckpt"


#create full set object (containing all data)
full_set =  rd.read_data_set(file_dir,file_name,"full_set",cut_output_max,cut_output_min,smooth_cut_range,N_in,N_out,y_ranges,train_ys_only_in_range,extend_data_artificially_in_ranges,do_not_consider_these_array_ids)

print("generate histogram with all data points")
pt.plot_histogram(full_set._y,y_ranges,energy,'linear','log',r"$\chi^2$","Number of points","./network_performance_plots/")


N_full_set = full_set._N
N_validation_set = 10000
N_training_set = N_full_set-N_validation_set

full_indices = np.random.permutation(N_full_set)

print("#########################################################################")
print("Starting SCYNET")
print("#########################################################################")
print("energy "+str(energy)+" TeV")
print("Extend data artificially "+ str(extend_data_artificially))
print("Sequence learning "+str(sequence_learning))


print("#########################################################################")
print("Finished reading in the full set with "+str(N_full_set)+" points")
print("Data structure")
print("x = "+str(full_set._x))
print("y = "+str(full_set._y))


training_indices = full_indices[:N_training_set] #indices for training set
validation_indices = full_indices[N_training_set:N_training_set+N_validation_set] #indices for validation set	


# create training set and validation set objects
training_set = rd.read_data_set(file_dir,file_name,"training_set",cut_output_max,cut_output_min,smooth_cut_range,N_in,N_out,y_ranges,train_ys_only_in_range,extend_data_artificially_in_ranges,do_not_consider_these_array_ids,full_set=full_set,subset=training_indices)
validation_set = rd.read_data_set(file_dir,file_name,"validation_set",cut_output_max,cut_output_min,smooth_cut_range,N_in,N_out,y_ranges,train_ys_only_in_range,extend_data_artificially_in_ranges,do_not_consider_these_array_ids,full_set=full_set,subset=validation_indices )


print("#########################################################################")
print("Finished generating trainig set with "+str(training_set._N)+" points")


print("#########################################################################")
print("Finished generating validation set with "+str(validation_set._N)+" points")



################################################################
#test performance of NN against a nearest neighbour interpolator
################################################################


### Nearest Neighbour interpolation for comparison with NN performance ####
interpol_near = ip_scipy.NearestNDInterpolator(training_set._x,training_set._y) 

y_interpol_near = interpol_near(validation_set._x)

error_validation_data_nearest_neighbour = np.subtract(y_interpol_near,validation_set._y)

total_error_validation_data_nearest_neighbour = np.sum(np.absolute(error_validation_data_nearest_neighbour))

mean_total_error_validation_data_nearest_neighbour = total_error_validation_data_nearest_neighbour/validation_set._N

rel_error_validation_data_nearest_neighbour = np.absolute(error_validation_data_nearest_neighbour)/validation_set._y

total_rel_error_validation_data_nearest_neighbour = np.sum(rel_error_validation_data_nearest_neighbour)

classification=1.0 # counts how many points have been classified with an error smaller than classification=1.0
classified_correct_validation_data_nearest_neighbour = np.sum((np.sign(-np.absolute(error_validation_data_nearest_neighbour)+classification)+1.0)*0.5)


#calculting everything for the ranges

y_interpol_near_ranges = [None]*len(y_ranges)
error_validation_data_nearest_neighbour_ranges = [None]*len(y_ranges)
total_error_validation_data_nearest_neighbour_ranges = np.array([0.0]*len(y_ranges))
mean_total_error_validation_data_nearest_neighbour_ranges = np.array([0.0]*len(y_ranges))
rel_error_validation_data_nearest_neighbour_ranges = [None]*len(y_ranges)
total_rel_error_validation_data_nearest_neighbour_ranges = np.array([0.0]*len(y_ranges))
classified_correct_validation_data_nearest_neighbour_ranges = np.array([0.0]*len(y_ranges))

for i in range(0,len(y_ranges)):
    y_interpol_near_ranges[i] = interpol_near(validation_set._xr[i])
    error_validation_data_nearest_neighbour_ranges[i] = np.subtract(y_interpol_near_ranges[i],validation_set._yr[i])
    total_error_validation_data_nearest_neighbour_ranges[i] = np.sum(np.absolute(error_validation_data_nearest_neighbour_ranges[i]))
    mean_total_error_validation_data_nearest_neighbour_ranges[i] = total_error_validation_data_nearest_neighbour_ranges[i]/validation_set._Nr[i]
    rel_error_validation_data_nearest_neighbour_ranges[i] = np.absolute(error_validation_data_nearest_neighbour_ranges[i])/validation_set._yr[i]
    total_rel_error_validation_data_nearest_neighbour_ranges[i] = np.sum(rel_error_validation_data_nearest_neighbour_ranges[i])
    classified_correct_validation_data_nearest_neighbour_ranges[i] = np.sum((np.sign(-np.absolute(error_validation_data_nearest_neighbour_ranges[i])+classification)+1.0)*0.5)



print("#########################################################################")
print("Nearest neighbour interpolator of validation set")

print("total error ")
print(total_error_validation_data_nearest_neighbour)
print("total error in ranges:")
print(total_error_validation_data_nearest_neighbour_ranges)
print("classified correctly within +-"+ str(classification))
print(classified_correct_validation_data_nearest_neighbour)
print("classified correctly within +-"+ str(classification))
print(classified_correct_validation_data_nearest_neighbour_ranges)
print("------------------------------------")
print("mean total error")
print(mean_total_error_validation_data_nearest_neighbour)
print("mean total error in ranges:")
print(mean_total_error_validation_data_nearest_neighbour_ranges)
print(" classified correctly within +-"+ str(classification)+" / number of total points")
print(classified_correct_validation_data_nearest_neighbour /validation_set._N)
print("classified correctly within +-"+ str(classification)+ " / number of points in ranges ")
print([classified_correct_validation_data_nearest_neighbour_ranges[i]/validation_set._Nr[i] for i in range(0,len(y_ranges))])
print("------------------------------------")





#################################
# Setting up the neural network 
#################################

# In the following we chose parameters for the neural network that have to be found by a large hypterparameter optimization

N_hlayer = 4 # number of hidden layer

N_epochs = 60 # number of training epochs

#for sequence learning
train_range=[-1]*N_epochs #train only with data in all ranges
if sequence_learning == "True":
    train_range=[-1,-1,0,-1,2,-1,3,-1] # first train two times with all data, then train only with data in 0th range, then again use full data set follwing by using only data in second range and so on...


#variables for adamOptimizer
learning_rat = 0.001
bet1 = 0.9# default 0.9
bet2 = 0.999# default 0.999
eps = 1e-08 #default 1e-08

	
batch_size = 500
lmda = 0.000001 #regularization parameter
cost = "quadratic" #quadratic_clever, quadratic, cross 
exp_damping = 0.0 # 0.0 means no exponential damping in the cost function

dropout_keep_prob=[1.0,1.0,1.0,1.0,1.0] #dropout keep probabilities in the layers. Iff set to 1 -> no dropout
N = [300]*(N_hlayer+2) # number of neurons in the layers
N[0] = N_in
N[-1] = N_out
activation_functions=["tanh"]*(N_hlayer+1) #activation functions in the layers


print("#########################################################################")
print("Start setting up the network")
print("Network parameters")
print("Input neurons "+str(N_in))
print("Output neurons "+str(N_out))
print("Number hidden layers "+str(N_hlayer))
print("Number of neurons in each hidden layer "+ str(N))
print("Activation functions "+str(activation_functions))
print("Variables for Adam optimizer")
print("Learning rate "+str(learning_rat))
print("bet1 "+str(bet1))
print("bet2 "+str(bet2))
print("eps "+str(eps))
print("batch size "+str(batch_size))
print("Regularization parameter "+str(lmda))
print("Cost function "+cost)
print("Exp damping "+str(exp_damping))
print("Dropout "+str(dropout_keep_prob))


with tf.Graph().as_default():

	#placeholders for dropout
    dropout_placeholder_1=tf.placeholder(tf.float32)
    dropout_placeholder_2=tf.placeholder(tf.float32)
    dropout_placeholder_3=tf.placeholder(tf.float32)
    dropout_placeholder_4=tf.placeholder(tf.float32)
    dropout_placeholder_5=tf.placeholder(tf.float32)
    dropout_placeholders=[dropout_placeholder_1,dropout_placeholder_2,dropout_placeholder_3,dropout_placeholder_4,dropout_placeholder_5]# for maximal 5 hidden layers

    x = tf.placeholder(tf.float32,shape=(None,N_in)) #don't take the shape=(batch_size,N1) argument, because we need this for different batch sizes
    w=[] # weight matrices
    b=[] # biases
    a=[x] # activations / outputs
    for i in range(0,len(N)-1):
        w.append(tf.Variable(tf.random.normal([N[i], N[i+1]],mean=0.0,stddev=1.0/math.sqrt(N[i]*1.0)),name="w_"+str(i))) #weights[0]=W2
        b.append(tf.Variable(tf.random.normal([N[i+1]]),name="b_"+str(i))) # biases[0]=b2
        if activation_functions[i]=="sigmoid":
            a.append(tf.sigmoid(tf.matmul(a[i], w[i]) + b[i])) # a[0]=a1=x,  a[1]=a2
        elif activation_functions[i]=="relu":
            a.append(tf.nn.relu(tf.matmul(a[i], w[i]) + b[i]))
        elif activation_functions[i]=="linear":
            a.append(tf.matmul(a[i], w[i]) + b[i])
        elif activation_functions[i]=="tanh":
            a.append(tf.tanh(tf.matmul(a[i], w[i]) + b[i]))

        if i<len(N)-2:# do not apply dropout to last layer
            a[-1]=tf.nn.dropout(a[-1],dropout_placeholders[i])

    y = a[len(N)-1] #output of the network as a function of all parameters. Note y=aN =oL
    y_ = tf.placeholder(tf.float32,shape=(None,N_out)) #  ,shape=(batch_size,N_out)

    #initializing cost functions, the total cost function is "cost_function_complete".
    cost_function_second_components = None
    cost_function_first_components = None
    cost_function_first = None
    cost_function_second = None
    cost_function_first_plus_second = None
    cost_function_complete = None

    #for exponential damping
    damping = -5.0/cut_output_max*exp_damping

    if cost=="quadratic" :
        if damping != 0.0:
            cost_function_complete = tf.scalar_mul(1.0/(N_training_set*2.0),tf.reduce_sum(tf.squared_difference(y,y_)*tf.exp(y_*damping)))
        else:
            cost_function_complete = tf.scalar_mul(1.0/(N_training_set*2.0),tf.reduce_sum(tf.squared_difference(y,y_)))

    c = 0.001
    sigma_ys = 1.0/(training_set._y_mod_max[0]*training_set._y_stddev[0]) #in the case of the chi2 we do not have an error for the chi2, so we artificially set the error to one

    m = sigma_ys * c
    if cost=="quadratic_clever":
        #in the case of quadratic_clever I did not include exponential damping
        cost_function_complete = tf.reduce_sum( tf.maximum(tf.minimum(m*(y-y_),c*(sigma_ys*sigma_ys)),0.0)+ tf.maximum(tf.minimum(-m*(y-y_),c*sigma_ys*sigma_ys),0.0)
		                         +(tf.square((y-y_)+m*0.5-sigma_ys)-m*m*0.25)*(tf.sign((y-y_)-sigma_ys)+1.0)*0.5 
		                         +(tf.square(-(y-y_)+m*0.5-sigma_ys)-m*m*0.25)*(tf.sign(-(y-y_)-sigma_ys)+1.0)*0.5   )

    elif cost=="cross":
        #There can be numerical problems with the logarithm!
        cost_function_first_components=tf.multiply((y_+1.0)*0.5,tf.log((y+1.0)*0.5))
        cost_function_second_components=tf.muliply( 1.0 - (y_+1.0)*0.5   ,  tf.log( 1.0 - (y+1.0)*0.5    ) )# Attention! One has to write 1- y. otherwise it does not work!

        if damping!=0.0:
            cost_function_first=tf.reduce_sum(cost_function_first_components*tf.exp(y_*damping))
            cost_function_second=tf.reduce_sum(cost_function_second_components*tf.exp(y_*damping))
        else:
            cost_function_first=tf.reduce_sum(cost_function_first_components)
            cost_function_second=tf.reduce_sum(cost_function_second_components)

        cost_function_first_plus_second=tf.add(cost_function_first,cost_function_second)
        cost_function_complete= tf.scalar_mul(-1.0/(N_training_set*1.0),cost_function_first_plus_second)

    #add regularization
    if lmda!=0.0:
        for i in range(0,len(w)):
            cost_function_complete=tf.add(cost_function_complete,tf.scalar_mul(lmda/(N_training_set*2.0),tf.reduce_sum(tf.square(w[i]))))

    error = y-y_ #error between desired and outputted values
    abs_error = tf.abs(y-y_)
    total_error = tf.reduce_sum(tf.abs(y-y_))

    l_rate = tf.placeholder(tf.float32) #placeholder for the learning rate

    train_step = tf.train.AdamOptimizer(learning_rate=l_rate,beta1=bet1, beta2=bet2,epsilon=eps).minimize(cost_function_complete)
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    #launch the graph
    sess = tf.Session()
    sess.run(init)

    # If we want to use a pre-trained net. Restore variables from disk. 
    if input_net != "":
        saver.restore(sess, input_net)
    else:
        print("no input net specified.")

    N_training_batch = int(training_set._N/batch_size) #rounds to samllest integer


    cost_training_data = [0.0]*N_epochs #cost function after each training epoch
    out_mod_validation_data = [0.0]*N_epochs # output of net, when inputting x_mod of validation data, saved after each epoch
    error_mod_validation_data = [0.0]*N_epochs # error between x_mod_desired and x_mod_outputted
    total_error_mod_validation_data = [0.0]*N_epochs #total error between x_mod_des and x_mod_outputted

    reduced = 0 #number of times where the learning rate was reduced
    for i in range(0,N_epochs):
        if train_range[i] == -1: #train with full training set
            print("Train with full training set")
            for j in range(0,N_training_batch):
                batch_xs, batch_ys, epochs_completed = training_set.next_batch(batch_size)#always gives the modified x's and y's. If one does not want to modify them the function has to be set to identity 
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, dropout_placeholder_1: dropout_keep_prob[0], dropout_placeholder_2: dropout_keep_prob[1],
			                                        dropout_placeholder_3: dropout_keep_prob[2], dropout_placeholder_4: dropout_keep_prob[3],dropout_placeholder_5: dropout_keep_prob[4],
			                                        l_rate : learning_rat}) #makes it possible to change the initial learning rate
        elif train_range[i] != -1: #train only with training data in specific range
            print("Train only with range: "+ str(train_range[i]))
            for j in range(0,training_set._Nr[train_range[i]]/batch_size):
                batch_xs, batch_ys, epochs_completed = training_set.next_batch_r(batch_size,train_range[i])#always gives the modified x's and y's. If one does not want to modifie them the function has to be set to identity 
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, dropout_placeholder_1: dropout_keep_prob[0], dropout_placeholder_2: dropout_keep_prob[1],
			                                        dropout_placeholder_3: dropout_keep_prob[2], dropout_placeholder_4: dropout_keep_prob[3],dropout_placeholder_5: dropout_keep_prob[4],
			                                        l_rate : learning_rat*1.0}) #when training with one specific range reduce learning rate by 0.05

        cost_training_data[i] = sess.run(cost_function_complete, feed_dict={
			x: training_set._x_mod, y_: training_set._y_mod, dropout_placeholder_1: 1.0, dropout_placeholder_2: 1.0, dropout_placeholder_3: 1.0,dropout_placeholder_4: 1.0,dropout_placeholder_5: 1.0})
        out_mod_validation_data[i] = sess.run(y, feed_dict={
			x: validation_set._x_mod, dropout_placeholder_1: 1.0, dropout_placeholder_2: 1.0, dropout_placeholder_3: 1.0,dropout_placeholder_4: 1.0,dropout_placeholder_5: 1.0})
        error_mod_validation_data[i] = sess.run(error , feed_dict={
			x: validation_set._x_mod,y_: validation_set._y_mod, dropout_placeholder_1: 1.0, dropout_placeholder_2: 1.0, dropout_placeholder_3: 1.0,dropout_placeholder_4: 1.0,dropout_placeholder_5: 1.0})
        total_error_mod_validation_data[i]=sess.run(total_error , feed_dict={
			x: validation_set._x_mod,y_: validation_set._y_mod, dropout_placeholder_1: 1.0, dropout_placeholder_2: 1.0, dropout_placeholder_3: 1.0,dropout_placeholder_4: 1.0,dropout_placeholder_5: 1.0})

        print("epochs completed: "+str(i))
        print("cost training data: "+ str(cost_training_data[i]))
        print("total error mod validation data: "+str(total_error_mod_validation_data[i]))
        print("output for x_mod validation as input")
        print(out_mod_validation_data[i])


        #reduce learning rate. check after 10 epochs if the slope of a fitted line is larger than some value
        if i % 10 == 0 and i>=20: #check every 10th epoch for saturation and if the learning rate will be decreased.
            z = 0.0
            n = 0.0
            mean_y = np.mean(np.array(total_error_mod_validation_data)[i-9:i+1])
            mean_x = (i*10.0-45.0)/10.0 # 1/10 * sum (i-j), j=0,...,9

            for j in range(0,10):
                z = z+(total_error_mod_validation_data[i-j]-mean_y)*(i-j-mean_x)
                n = n+(i-j-mean_x)**2.0
            m=z/n
            print("m "+ str(m)) #slope of a line which is fitted trough the last 10 points

            if m>-5.0 :# one has to set this threhold according to the transformations one does on the data.
                learning_rat = learning_rat*0.5
                reduced = reduced + 1
                print("learning rate reduced by a factor of 0.5")

        print("------------------------------------------------------------------")

    print("learning rate was reduced "+ str(reduced) + " times by a factor of 0.5")

    # Save the variables to disk.
    if output_net!="":
        save_path = saver.save(sess, output_net)
        print("Model saved in file: %s" % save_path)
    else:
        print("net was not saved to disk!")

    out_validation_data = [0]*N_epochs # output of net, when inputting x_mod of validation data and making the normalization backwards, saved after each epoch
    error_validation_data = [0.0]*N_epochs
    total_error_validation_data = np.array([0.0]*N_epochs)
    mean_total_error_validation_data = np.array([0.0]*N_epochs)
    rel_error_validation_data = [0.0]*N_epochs
    total_rel_error_validation_data = np.array([0.0]*N_epochs)
    classification_validation_data = np.array([0.0]*N_epochs)

    #make the transformation on the outputs backwards
    for i in range(0,N_epochs):
        out_validation_data[i] = np.copy(out_mod_validation_data[i])
        
        rd.apply_inverse_function_to_outputs(out_mod_validation_data[i],out_validation_data[i],full_set._y_max,full_set._y_min,full_set._y_mean,full_set._y_stddev,full_set._y_mod_max)# second argument will be changed!

        error_validation_data[i] = np.subtract(out_validation_data[i],validation_set._y)
        total_error_validation_data[i] = np.sum(np.absolute(error_validation_data[i]))
        mean_total_error_validation_data[i] = total_error_validation_data[i]/validation_set._N

        rel_error_validation_data[i] = np.absolute(error_validation_data[i])/validation_set._y
        total_rel_error_validation_data[i] = np.sum(np.absolute(error_validation_data[i])/validation_set._y)

        classification_validation_data[i] =  np.sum((np.sign(-np.absolute(error_validation_data[i])+classification)+1.0)*0.5)


    print("output (for x_mod_validation as input)")
    print(out_mod_validation_data[-1])
    print("desired output y_mod_validation")
    print(validation_set._y_mod)
    print("total error between them")
    print(total_error_mod_validation_data)
    print("--------------------------------------------")
    print("--------------------------------------------")
    print("output (for x_validation as input)")
    print(out_validation_data[-1])
    print("desired output y_validation")
    print(validation_set._y)
    print("total error validation data")
    print(total_error_validation_data)
    print("total rel error validation data")
    print(total_rel_error_validation_data)
    print("classification validation data")
    print(classification_validation_data)
    print("--------------------------------------------")
    print("mean total error validation data")
    print(mean_total_error_validation_data)
    print("total rel error validation data/N_val data")
    print(total_rel_error_validation_data/validation_set._N)
    print("classification validation data/ N_val data")
    print(classification_validation_data/validation_set._N)
    print("--------------------------------------------")
    print("--------------------------------------------")

    #now do everything for the y ranges seperately
    error_validation_data_ranges = [None]*N_epochs
    total_error_validation_data_ranges = np.ndarray(shape=(N_epochs,len(y_ranges)))
    mean_total_error_validation_data_ranges = np.ndarray(shape=(N_epochs,len(y_ranges)))
    rel_error_validation_data_ranges = [None]*N_epochs
    total_rel_error_validation_data_ranges = np.ndarray(shape=(N_epochs,len(y_ranges)))
    classification_validation_data_ranges = np.ndarray(shape=(N_epochs,len(y_ranges)))

    for m in range(0,N_epochs):
        error_validation_data_ranges[m] = [np.ndarray(shape=(validation_set._Nr[u],N_out)) for u in range(0,len(y_ranges))]
        rel_error_validation_data_ranges[m] = [np.ndarray(shape=(validation_set._Nr[u],N_out)) for u in range(0,len(y_ranges))]
        k=[0]*len(y_ranges)
        for i in range(0,validation_set._N):
            for j in range(0,len(y_ranges)):
                if y_ranges[j][0]<validation_set._y[i][0] and validation_set._y[i][0]<=y_ranges[j][1]:
                    error_validation_data_ranges[m][j][k[j]][0] = error_validation_data[m][i][0]
                    rel_error_validation_data_ranges[m][j][k[j]][0] = rel_error_validation_data[m][i][0]
                    k[j]=k[j]+1

    for m in range(0,N_epochs):
        for j in range(0,len(y_ranges)):
            total_error_validation_data_ranges[m][j] = np.sum(np.absolute(error_validation_data_ranges[m][j]))
            mean_total_error_validation_data_ranges[m][j] = total_error_validation_data_ranges[m][j]/validation_set._Nr[j]
            total_rel_error_validation_data_ranges[m][j] = np.sum(np.absolute(rel_error_validation_data_ranges[m][j]))
            classification_validation_data_ranges[m][j] = np.sum((np.sign(-np.absolute(error_validation_data_ranges[m][j])+classification)+1.0)*0.5)

    print("total_error_validation_data_ranges")
    print(total_error_validation_data_ranges)
    print("total_rel_error_validation_data_ranges")
    print(total_rel_error_validation_data_ranges)
    print("classification_validation_data_ranges in +- "+str(classification))
    print(classification_validation_data_ranges)
    print("--------------------------------------------")
    print("mean_total_error_validation_data_ranges")
    print(mean_total_error_validation_data_ranges)
    print("total_rel_error_validation_data_ranges/ number in range")
    print([total_rel_error_validation_data_ranges[i]/(validation_set._Nr*1.0) for i in range(0,N_epochs)])
    print("classification_validation_data_ranges/ number in range in +- "+str(classification))
    print([classification_validation_data_ranges[i]/(validation_set._Nr*1.0) for i in range(0,N_epochs)])
    print("--------------------------------------------")
    print("--------------------------------------------")



# save all important data

output_dir = "./network_performance_data/"

np.save(output_dir+'total_error_validation_data_'+energy+'TeV.npy', total_error_validation_data)
np.save(output_dir+'mean_total_error_validation_data_'+energy+'TeV.npy',mean_total_error_validation_data)
np.save(output_dir+'total_rel_error_validation_data_'+energy+'TeV.npy',total_rel_error_validation_data)
np.save(output_dir+'classification_validation_data_'+energy+'TeV.npy',classification_validation_data)

np.save(output_dir+'total_error_validation_data_ranges_'+energy+'TeV.npy', total_error_validation_data_ranges)
np.save(output_dir+'mean_total_error_validation_data_ranges_'+energy+'TeV.npy',mean_total_error_validation_data_ranges)
np.save(output_dir+'total_rel_error_validation_data_ranges_'+energy+'TeV.npy',total_rel_error_validation_data_ranges)
np.save(output_dir+'classification_validation_data_ranges_'+energy+'TeV.npy',classification_validation_data_ranges)



np.save(output_dir+'total_error_validation_data_nearest_neighbour_'+energy+'TeV.npy',np.array(total_error_validation_data_nearest_neighbour))
np.save(output_dir+'mean_total_error_validation_data_nearest_neighbour_'+energy+'TeV.npy',np.array(mean_total_error_validation_data_nearest_neighbour))
np.save(output_dir+'total_rel_error_validation_data_nearest_neighbour_'+energy+'TeV.npy', np.array(total_rel_error_validation_data_nearest_neighbour) )
np.save(output_dir+'classified_correct_validation_data_nearest_neighbour_'+energy+'TeV.npy',np.array(classified_correct_validation_data_nearest_neighbour))

np.save(output_dir+'total_error_validation_data_nearest_neighbour_ranges_'+energy+'TeV.npy',np.array(total_error_validation_data_nearest_neighbour_ranges))
np.save(output_dir+'mean_total_error_validation_data_nearest_neighbour_ranges_'+energy+'TeV.npy',np.array(mean_total_error_validation_data_nearest_neighbour_ranges))
np.save(output_dir+'total_rel_error_validation_data_nearest_neighbour_ranges_'+energy+'TeV.npy', np.array(total_rel_error_validation_data_nearest_neighbour_ranges) )
np.save(output_dir+'classified_correct_validation_data_nearest_neighbour_ranges_'+energy+'TeV.npy',np.array(classified_correct_validation_data_nearest_neighbour_ranges))

