#################################
# This code was written by Jan Schuette-Engel (schuette@physik.rwth-aachen.de)
# Use this code only with explicit permission.
# If you publish any results which are based on this code please cite the upcoming SCYNET paper
#################################

import readData_LHC_chi2 as rd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from random import randint
import random
from root_numpy import fill_hist

from ROOT import TCanvas, TH2F, TText, TF1 ,TH1D, TH2D, TFile, TLegend, TPaletteAxis
import ROOT

import tensorflow as tf

import math
import sys
import scipy.interpolate as ip_scipy

import os

import os.path

from array import array




# # # # # # # #
#Read in data #
#             #




file_dir="/home/js668623/neural_nets/pMSSM_11_data"
file_name="" #this will be set below



N_in=11
N_out=1


energy = "8" # 13 or 8

if energy == "8":
  h_2 = energy+"TeV_chi2_disjoint False False False 2.0 2.0 1.5" # best hyperparameters for the second scan at 8 TeV 
elif energy == "13":
  h_2 = energy+"TeV_chi2_disjoint False False False 2.0 2.0 1.0" # best hyperparameters for the second scan at 13 TeV, original last hyperparameter in string 1.0


h_2_str = h_2
h_2 = h_2.split( )
print "configuration from second hyperparameter scan "+str(h_2)


file_name = h_2[0]
print "file_name"
print file_name


train_ys_only_in_range = None #only targets in the range which will be specified with this variable will be considered

if energy == "8":
  train_ys_only_in_range = [0,100] # [0,200]: [0,50] , [40,120.0], [110.0,190], [180,200] ;;; [0,100] , [40,80], [70,95]
elif energy== "13":
  train_ys_only_in_range = [0,100] # [0,100], [0,60] , [55,100] , [90,190], [180,200] [0,200] # only targets in these ranges will be considered



cut_output_max = 100.0 # smooth cut (maximal value of the targets) # 200, 100
smooth_cut_range = 25.0 # cut range for the smooth cut. If set to -1 the cut will be hard. Do not set to zero for a hard cut. Set it -1 for a hard cut # 40, 25


cut_output_min = 0.0 # hard cut at cut_output_min, furthermore the backtransformed values of the network are never smaller than cut_output_min

if energy == "8":
  cut_output_min = 11.9
elif energy == "13":
  cut_output_min = 25.4

y_ranges = None # the errors will be given seperately for all ranges. This is important if one wants to investigate the RTLP



if energy == "8":
  if train_ys_only_in_range[0] == 0 and train_ys_only_in_range[1] == 50:
    y_ranges = [[-0.001,31.0],[31.0,34.0],[34.0,50]]
  elif train_ys_only_in_range[0] == 40 and train_ys_only_in_range[1] == 120:
    y_ranges = [[39.9999,70.0],[70.0,120.0001]]
  elif train_ys_only_in_range[0] == 110 and train_ys_only_in_range[1] == 190:
    y_ranges = [[109.9999,140.0],[140.0,190.0001]]
  elif train_ys_only_in_range[0] == 180 and train_ys_only_in_range[1] == 200:
    y_ranges = [[179.9999,195.0],[195.0,200.0001]]
  elif train_ys_only_in_range[0] == 0 and train_ys_only_in_range[1] == 200:
    y_ranges = [[-0.001,31.0],[31.0,34.0],[34.0,50],[50.0,120],[120,195.0],[195.0,200.0001]]
  elif train_ys_only_in_range[0] == 0 and train_ys_only_in_range[1] == 100:
    y_ranges = [[0.0,38.0],[38.0,42.0],[42.0,70.0],[70.0,95.0],[95.0,100.0]]
  elif train_ys_only_in_range[0] == 40 and train_ys_only_in_range[1] == 80:
    y_ranges = [[40.0,60.0],[60.0,80.0001]]
  elif train_ys_only_in_range[0] == 70 and train_ys_only_in_range[1] == 95:
    y_ranges = [[70.0,85.0],[85.0,95.0001]]
  elif train_ys_only_in_range[0] == 95 and train_ys_only_in_range[1] == 100:
    y_ranges = [[95.0,99.0],[99.0,100.0001]]
elif energy == "13":
  if train_ys_only_in_range[0] == 0 and train_ys_only_in_range[1] == 60:
    y_ranges = [[-0.001,46.5],[46.5,49.0],[49.0,60]]
  elif train_ys_only_in_range[0] == 55 and train_ys_only_in_range[1] == 100:
    y_ranges = [[54.9999,70.0],[70.0,100.0001]]
  elif train_ys_only_in_range[0] == 90 and train_ys_only_in_range[1] == 190:
    y_ranges = [[89.9999,140.0],[140.0,190.0001]]
  elif train_ys_only_in_range[0] == 180 and train_ys_only_in_range[1] == 200:
    y_ranges = [[179.9999,195.0],[195.0,200.0001]]
  elif train_ys_only_in_range[0] == 0 and train_ys_only_in_range[1] == 200:
    y_ranges = [[-0.001,46.5],[46.5,49.0],[49.0,60],[60.0,70.0],[70.0,100.0],[100.0,140.0],[140.0,180.0],[180.0,195.0],[195.0,200.0001]]
  elif train_ys_only_in_range[0] == 0 and train_ys_only_in_range[1] == 100:
    y_ranges = [[0.0,53.5],[53.5,56.0],[56.0,70],[70.0,95.0],[95.0,100.0]]


extend_data_artificially_in_ranges = [1]*len(y_ranges) #no artificial extension per default
if h_2[2] == "True":
  if energy== "8":
    if train_ys_only_in_range[0] == 0 and train_ys_only_in_range[1] == 100:
      extend_data_artificially_in_ranges = [3,1,1,3,1]
    elif train_ys_only_in_range[0] == 40 and train_ys_only_in_range[1] == 80:
      extend_data_artificially_in_ranges=[1,2]
    elif train_ys_only_in_range[0] == 70 and train_ys_only_in_range[1] == 95:
      extend_data_artificially_in_ranges=[1,2]
    elif train_ys_only_in_range[0] == 95 and train_ys_only_in_range[1] == 100:
      extend_data_artificially_in_ranges=[2,1]
    elif train_ys_only_in_range[0] == 0 and train_ys_only_in_range[1] == 200:
      extend_data_artificially_in_ranges=[3,1,2,3,3,1]
  elif energy== "13":
    if train_ys_only_in_range[0] == 0 and train_ys_only_in_range[1] == 60:
      extend_data_artificially_in_ranges = [2,1,2]
    elif train_ys_only_in_range[0] == 0 and train_ys_only_in_range[1] == 100:
      extend_data_artificially_in_ranges = [2,2,2,2,1]


use_extra_raw_detail_samples = h_2[1] # to test if the extra sampled data in rare target areas gives an improovement
do_not_consider_these_array_ids= [[-1]] # per default consider all array id's

if h_2[1] == "False":
  if energy == "8":
    do_not_consider_these_array_ids = [[125000,1100000000]]
    #do_not_consider_these_array_ids= [[100000000,200000000],[300000000,500000000],[600000000,1100000000]] #TODO: wieder einkommentieren
  elif energy == "13":
    do_not_consider_these_array_ids = [[-1]] #because there is no extra sampled data yet for 13 TeV


#for sequence learning
train_range=[-1] #train only with data in all ranges
if h_2[3] == "True":
  if energy== "8": 
    if train_ys_only_in_range[0] == 0 and train_ys_only_in_range[1] == 60:
      train_range=[-1,-1,0,-1,2]
    elif train_ys_only_in_range[0] == 55 and train_ys_only_in_range[1] == 100:
      train_range=[-1,-1,-1,1]
    elif train_ys_only_in_range[0] == 90 and train_ys_only_in_range[1] == 190:
      train_range=[-1,-1,-1,1]
    elif train_ys_only_in_range[0] == 190 and train_ys_only_in_range[1] == 200:
      train_range=[-1,-1,-1,0]
    elif train_ys_only_in_range[0] == 0 and train_ys_only_in_range[1] == 200:
      train_range=[-1,-1,-1,0] 
    elif train_ys_only_in_range[0] == 0 and train_ys_only_in_range[1] == 100:
      train_range=[-1,-1,0,-1,2,-1,3,-1]
    elif train_ys_only_in_range[0] == 40 and train_ys_only_in_range[1] == 80:
      train_range=[-1,-1,-1,1,-1]
    elif train_ys_only_in_range[0] == 70 and train_ys_only_in_range[1] == 95:
      train_range=[-1,-1,-1,1,-1]
    elif train_ys_only_in_range[0] == 95 and train_ys_only_in_range[1] == 100:
      train_range=[-1,-1,-1,0,-1]
    elif train_ys_only_in_range[0] == 0 and train_ys_only_in_range[1] == 200:
      train_range=[-1,-1,-1,0,-1,2,-1,3,-1,4,-1]
  if energy== "13":
    if train_ys_only_in_range[0] == 0 and train_ys_only_in_range[1] == 60:
      train_range=[-1,-1,0,-1,2]
    elif train_ys_only_in_range[0] == 55 and train_ys_only_in_range[1] == 100:
      train_range=[-1,-1,-1,1]
    elif train_ys_only_in_range[0] == 90 and train_ys_only_in_range[1] == 190:
      train_range=[-1,-1,-1,1]
    elif train_ys_only_in_range[0] == 190 and train_ys_only_in_range[1] == 200:
      train_range=[-1,-1,-1,0]
    elif train_ys_only_in_range[0] == 0 and train_ys_only_in_range[1] == 200:
      train_range=[-1,-1,-1,0] #TODO das muesste ich mir nochmal ueberlegen
    elif train_ys_only_in_range[0] == 0 and train_ys_only_in_range[1] == 100:
      train_range=[-1,-1,0,-1,-1,2,-1,3] # -1 means training with full range 


outputfolder = "/home/js668623/neural_nets/pMSSM_11_tensorflow/chi2/classifier_plus_net_for_each_range/picture_outputs/"+energy+"TeV_range_"+str(train_ys_only_in_range[0])+"_"+str(train_ys_only_in_range[1])

#outputfolder = "/home/js668623/neural_nets/pMSSM_11_tensorflow/chi2/classifier_plus_net_for_each_range/picture_outputs/"+energy+"TeV_range_"+str(train_ys_only_in_range[0])+"_"+str(train_ys_only_in_range[1])

if not os.path.isdir(outputfolder):
  print "make output folder"
  os.makedirs(outputfolder)


#TODO: naechste Zeile wieder aendern
#input_net = "./picture_outputs/13TeV/min=min_possible/13TeV_range_0_100_third_try/net.ckpt" #if "" no input net will be taken otherwise one has to specify the path to the input net, e.g. "./net.ckpt"
input_net = ""
output_net = outputfolder+"/net.ckpt"


validation_set_with_fixed_array_id = [[-1]] # this is default. Choosing this gives a randomly choosen validation set out of the full set. But sometimes one has to specify the array_id's of the validaton set

if energy == "8":
  print "use no fixed val set"
  #validation_set_with_fixed_array_id = [[200065000,300000000],[500000000,600000000]] # we choose this validation set, because there is no extra sampling around thiese points #TODO: wieder rein machen
elif energy == "13":
  validation_set_with_fixed_array_id = [[-1]] # -1 means a random validation set considering all array id's 


#create full set object
full_set =  rd.read_data_set(file_dir,file_name,"full_set",cut_output_max,cut_output_min,smooth_cut_range,N_in,N_out,y_ranges,train_ys_only_in_range,extend_data_artificially_in_ranges,do_not_consider_these_array_ids,validation_set_with_fixed_array_id)

#if the option for a fixed validation set is not given (e.g. validation_set_with_fixed_array_id = [[-1]]) then the validation subset will be determined with the index subset from below
N_full_set = full_set.get_N()
N_validation_set = 10000
N_training_set = N_full_set-N_validation_set

full = range(0,N_full_set)

random.shuffle(full)

training_subset = full[:N_training_set]#indices for training set
validation_subset = full[N_training_set:N_training_set+N_validation_set]#indices for validation set	


# create training set and validation set objects
training_set = rd.read_data_set(file_dir,file_name,"training_set",cut_output_max,cut_output_min,smooth_cut_range,N_in,N_out,y_ranges,train_ys_only_in_range,extend_data_artificially_in_ranges,do_not_consider_these_array_ids,validation_set_with_fixed_array_id,
                                full_set=full_set,subset=training_subset)
validation_set = rd.read_data_set(file_dir,file_name,"validation_set",cut_output_max,cut_output_min,smooth_cut_range,N_in,N_out,y_ranges,train_ys_only_in_range,extend_data_artificially_in_ranges,do_not_consider_these_array_ids,validation_set_with_fixed_array_id,
                                  full_set=full_set,subset=validation_subset )



print "------------------"
print "full_data_set:"
print "x (inputs)"
print full_set.get_x()
print "y (outputs)"
print full_set.get_y()
print "x_mod"
print full_set.get_x_mod()
print "y_mod"
print full_set.get_y_mod()
print "N"
print full_set.get_N()
print "Nr"
print full_set.get_Nr_tot()
print "------------------"
print "training_data_set:"
print "x (inputs)"
print training_set.get_x()
print "y (outputs)"
print training_set.get_y()
print "x_mod"
print training_set.get_x_mod()
print "y_mod"
print training_set.get_y_mod()
print "N"
print training_set.get_N()
print "Nr"
print training_set.get_Nr_tot()
print "N_ea"
print training_set.get_N_ea()
print "Nr_ea"
print training_set.get_Nr_ea_tot()
print "------------------"
print "evaluation_data_set:"
print "x (inputs)"
print validation_set.get_x()
print "y (outputs)"
print validation_set.get_y()
print "x_mod"
print validation_set.get_x_mod()
print "y_mod"
print validation_set.get_y_mod()
print "N"
print validation_set.get_N()
print "Nr"
print validation_set.get_Nr_tot()
print "------------------"
print "x_max"
print full_set.get_x_max()
print "x_min"
print full_set.get_x_min()
print "y_max"
print full_set.get_y_max()
print "y_min"
print full_set.get_y_min()
print "x_mean"
print full_set.get_x_mean()
print "y_mean"
print full_set.get_y_mean()
print "x_stddev"
print full_set.get_x_stddev()
print "y_stddev"
print full_set.get_y_stddev()
print "x_mod_max"
print full_set.get_x_mod_max()
print "y_mod_max"
print full_set.get_y_mod_max()
print "------------------"
print "y_ranges"
print y_ranges
print "extend data artificially in ranges:"
print extend_data_artificially_in_ranges 
print "use_extra_raw_detail_samples"
print use_extra_raw_detail_samples
print do_not_consider_these_array_ids
print "sequence learning"
print train_range
print "------------------"




# # # # # # # # # # # # # # # # #
#other interpolation techniques #
# # # # # # # # # # # # # # # # # 


#scipy.interpolate library

#interpol_lin = ip_scipy.LinearNDInterpolator(training_set.get_x(),training_set.get_y()) # this algorithm is not stable and gives nan sometimes
#y_interpol_lin = interpol_lin(validation_set.get_x())
#error_interpol_lin=np.sum(np.absolute(np.subtract(y_interpol_lin,validation_set.get_y())))
#print "total error with LinearNDInterpolator"
#print error_interpol_lin


### Nearest Neighbour interpolation ####
interpol_near = ip_scipy.NearestNDInterpolator(training_set.get_x(),training_set.get_y()) #this algorithm is stable

y_interpol_near = interpol_near(validation_set.get_x())

error_validation_data_nearest_neighbour = np.subtract(y_interpol_near,validation_set.get_y())#.reshape((1,validation_set.get_N()))[0]

total_error_validation_data_nearest_neighbour = np.sum(np.absolute(error_validation_data_nearest_neighbour))

rel_error_validation_data_nearest_neighbour = np.absolute(error_validation_data_nearest_neighbour)/validation_set.get_y()

total_rel_error_validation_data_nearest_neighbour = np.sum(rel_error_validation_data_nearest_neighbour)

classification=1.0
classified_correct_validation_data_nearest_neighbour = np.sum((np.sign(-np.absolute(error_validation_data_nearest_neighbour)+classification)+1.0)*0.5)


#calculting everything for the ranges

y_interpol_near_ranges = [None]*len(y_ranges)

for i in range(0,len(y_ranges)):
	y_interpol_near_ranges[i] = interpol_near(validation_set.get_xr(i))


error_validation_data_nearest_neighbour_ranges = [None]*len(y_ranges)

for i in range(0,len(y_ranges)):
	error_validation_data_nearest_neighbour_ranges[i] = np.subtract(y_interpol_near_ranges[i],validation_set.get_yr(i))#.reshape((1,validation_set.get_N()))[0]


total_error_validation_data_nearest_neighbour_ranges = np.array([0.0]*len(y_ranges))

for i in range(0,len(y_ranges)):
	total_error_validation_data_nearest_neighbour_ranges[i] = np.sum(np.absolute(error_validation_data_nearest_neighbour_ranges[i]))


rel_error_validation_data_nearest_neighbour_ranges = [None]*len(y_ranges)

for i in range(0,len(y_ranges)):
	rel_error_validation_data_nearest_neighbour_ranges[i] = np.absolute(error_validation_data_nearest_neighbour_ranges[i])/validation_set.get_yr(i)


total_rel_error_validation_data_nearest_neighbour_ranges = np.array([0.0]*len(y_ranges))

for i in range(0,len(y_ranges)):
	total_rel_error_validation_data_nearest_neighbour_ranges[i] = np.sum(rel_error_validation_data_nearest_neighbour_ranges[i])



classified_correct_validation_data_nearest_neighbour_ranges = np.array([0.0]*len(y_ranges))

for i in range(0,len(y_ranges)):
	classified_correct_validation_data_nearest_neighbour_ranges[i] = np.sum((np.sign(-np.absolute(error_validation_data_nearest_neighbour_ranges[i])+classification)+1.0)*0.5)


print "Nearest neighbour interpolator"

print "total error "
print total_error_validation_data_nearest_neighbour
print "total error in ranges:"
print total_error_validation_data_nearest_neighbour_ranges
print " classification accuracy for +-"+ str(classification)
print classified_correct_validation_data_nearest_neighbour 
print "classification accuracy in ranges for +-"+ str(classification)
print classified_correct_validation_data_nearest_neighbour_ranges
print "------------------------------------"
print "total error/ total number of elements in validation set "
print total_error_validation_data_nearest_neighbour/validation_set.get_N()
print "total error in ranges/elements in ranges:"
print [total_error_validation_data_nearest_neighbour_ranges[i]/validation_set.get_Nr(i) for i in range(0,len(y_ranges))]
print " classification accuracy for +-"+ str(classification)
print classified_correct_validation_data_nearest_neighbour /validation_set.get_N()
print "classification accuracy in ranges for +-"+ str(classification)
print [classified_correct_validation_data_nearest_neighbour_ranges[i]/validation_set.get_Nr(i) for i in range(0,len(y_ranges))]
print "------------------------------------"


#TODO: fuege noch einen anderen interpolator hinzu!





# # # # # # # # # # # # #
#setting up the network #
# # # # # # # # # # # # #


#read hyperparameters from hyperparameters_to_scan
"""
array_id=int(os.environ['LSB_JOBINDEX'])
fr=open("./hyperparameters_to_scan","r") 
lines=fr.readlines()
fr.close()
line=lines[array_id]
line_str=line
line=line.split( )
print line
"""

#give manually the best found hyperparameter configuration
array_id = 0 
#best result from first hyperparameter scan
h_1 = '4 150 quadratic_clever 1.0 500 1e-05 0.001 1.0 1.0 tanh' 
h_1_str = h_1
h_1 = h_1.split( )
print "hyperparameter configuration from first scan:"
print h_1_str



N_epochs = 2 # number of training epochs #TODO: wieder wegmachen

#variables for adamOptimizer
learning_rat =  float(h_1[6]) #TODO: wieder wegmachen
bet1=0.9# default 0.9
bet2=0.999# default 0.999
eps=1e-08 #default 1e-08

	
batch_size = int(float(h_1[4])*float(h_2[6]))
lmda=float(h_1[5]) #regularization parameter
cost = h_1[2] #"quadratic_clever" # eigentlich quadratic cost in hyperparameter scan gefunden, aber zu dem Zeitpunkt habe ich clever quadratic cost noch nicht probiert, im nachhinein hat sich clever quadratic cost als besser herausgestellt.

exp_damping = 0.0 # 0.0 means no exponential damping

dropout_keep_prob=[]
N=[]
activation_functions=[]

if int(h_1[0])==2: #two hidden layer
	dropout_keep_prob=[float(h_1[7]),float(h_1[8]),1.0,1.0,1.0]# dropout will only be applied to the hidden layers
	N=[N_in,int(h_1[1]),int(h_1[1]),N_out]
	activation_functions=["tanh","tanh",str(h_1[9])]

if int(h_1[0])==3: #3 hidden layer
	dropout_keep_prob=[float(h_1[7]),float(h_1[8]),float(h_1[7]),1.0,1.0]# dropout will only be applied to the hidden layers
	N=[N_in,int(h_1[1]),int(h_1[1]),int(h_1[1]),N_out]
	activation_functions=["tanh","tanh","tanh",str(h_1[9])]

if int(h_1[0])==4: #4 hidden layer
	dropout_keep_prob=[float(h_1[7]),float(h_1[8]),float(h_1[7]),float(h_1[8]),1.0]# dropout will only be applied to the hidden layers
	N=[N_in,int(float(h_1[1])*float(h_2[4])),int(float(h_1[1])*float(h_2[5])),int(float(h_1[1])*float(h_2[5])),int(float(h_1[1])*float(h_2[5])),N_out]
	activation_functions=["tanh","tanh","tanh","tanh",str(h_1[9])]

if int(h_1[0])==5: #5 hidden layer
	dropout_keep_prob=[float(h_1[7]),float(h_1[8]),float(h_1[7]),float(h_1[8]),float(h_1[7])]# dropout will only be applied to the hidden layers
	N=[N_in,int(float(h_1[1])*float(h_2[4])),int(float(h_1[1])*float(h_2[5])),int(float(h_1[1])*float(h_2[5])),int(float(h_1[1])*float(h_2[5])),int(float(h_1[1])*float(h_2[5])),N_out]
	activation_functions=["tanh","tanh","tanh","tanh","tanh",str(h_1[9])]


with tf.Graph().as_default():

	#placeholders for dropout
	dropout_placeholder_1=tf.placeholder(tf.float32)
	dropout_placeholder_2=tf.placeholder(tf.float32)
	dropout_placeholder_3=tf.placeholder(tf.float32)
	dropout_placeholder_4=tf.placeholder(tf.float32)
	dropout_placeholder_5=tf.placeholder(tf.float32)
	dropout_placeholders=[dropout_placeholder_1,dropout_placeholder_2,dropout_placeholder_3,dropout_placeholder_4,dropout_placeholder_5]# for maximal 5 hidden layers


	x = tf.placeholder(tf.float32,shape=(None,N[0])) #don't take the shape=(batch_size,N1) argument, because we need this for different batch sizes
	w=[] #weight matrices
	b=[] #biases
	a=[x] #activations / outputs
	for i in range(0,len(N)-1):
		w.append(tf.Variable(tf.random_normal([N[i], N[i+1]],mean=0.0,stddev=1.0/math.sqrt(N[i]*1.0)),name="w_"+str(i))) #weights[0]=W2
		b.append(tf.Variable(tf.random_normal([N[i+1]]),name="b_"+str(i))) # biases[0]=b2
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


	y = a[len(N)-1] #y=aN =oL

	y_ = tf.placeholder(tf.float32,shape=(None,N_out)) #  ,shape=(batch_size,N_out)



	# # # # # # # # # # # # # #
	#initializing and training#
	#                         #

	cost_function_second_components=None
	cost_function_first_components=None

	cost_function_first=None
	cost_function_second=None

	cost_function_first_plus_second=None

	cost_function_complete=None

	#for exponential damping
	damping=-5.0/cut_output_max*exp_damping

	if cost=="quadratic" :
		if damping!=0.0:
			cost_function_complete = tf.scalar_mul(1.0/(N_training_set*2.0),tf.reduce_sum(tf.squared_difference(y,y_)*tf.exp(y_*damping)))
		else:
			cost_function_complete = tf.scalar_mul(1.0/(N_training_set*2.0),tf.reduce_sum(tf.squared_difference(y,y_)))

		if lmda!=0:
			for i in xrange(0,len(w)):
				cost_function_complete=tf.add(cost_function_complete,tf.scalar_mul(lmda/(N_training_set*2.0),tf.reduce_sum(tf.square(w[i]))))



	c = 0.001
	sigma_ys = 1.0/(training_set.get_y_mod_max()[0]*training_set.get_y_stddev()[0]) #in the case of the chi2 we do not have an error for the chi2, so we artificially set the error to one

	m= sigma_ys * c
	if cost=="quadratic_clever":
		#in the case of quadratic_clever I did not include exponential damping
		cost_function_complete = tf.reduce_sum( tf.maximum(tf.minimum(m*(y-y_),c*(sigma_ys*sigma_ys)),0.0)+ tf.maximum(tf.minimum(-m*(y-y_),c*sigma_ys*sigma_ys),0.0)
		                         +(tf.square((y-y_)+m*0.5-sigma_ys)-m*m*0.25)*(tf.sign((y-y_)-sigma_ys)+1.0)*0.5 
		                         +(tf.square(-(y-y_)+m*0.5-sigma_ys)-m*m*0.25)*(tf.sign(-(y-y_)-sigma_ys)+1.0)*0.5   )
		#Wenn man die theta funktion vermeiden will kann man alternativ auch sigmoids nehmen! 

		if lmda!=0:
                        for i in xrange(0,len(w)):
                                cost_function_complete=tf.add(cost_function_complete,tf.scalar_mul(lmda/(2.0),tf.reduce_sum(tf.square(w[i]))))


	elif cost=="cross":
		#es kann zu problemen mit dem ln kommen, wenn wegen rundungen -1 ausgegeben wird und dann der ln(0) ausgewertet wird.
		cost_function_first_components=tf.mul((y_+1.0)*0.5,tf.log((y+1.0)*0.5))
		cost_function_second_components=tf.mul( 1.0 - (y_+1.0)*0.5   ,  tf.log( 1.0 - (y+1.0)*0.5    ) )# Attention! One has to write 1- y. otherwise it does not work!

		if damping!=0.0:
			cost_function_first=tf.reduce_sum(cost_function_first_components*tf.exp(y_*damping))
			cost_function_second=tf.reduce_sum(cost_function_second_components*tf.exp(y_*damping))
		else:
			cost_function_first=tf.reduce_sum(cost_function_first_components)
			cost_function_second=tf.reduce_sum(cost_function_second_components)
		
		cost_function_first_plus_second=tf.add(cost_function_first,cost_function_second)
		
		cost_function_complete= tf.scalar_mul(-1.0/(N_training_set*1.0),cost_function_first_plus_second)
		
		if lmda!=0.0:
			for i in xrange(0,len(w)):
				cost_function_complete=tf.add(cost_function_complete,tf.scalar_mul(lmda/(N_training_set*2.0),tf.reduce_sum(tf.square(w[i]))))


	error = y-y_ #error between desired and outputted values
	abs_error = tf.abs(y-y_)
	total_error = tf.reduce_sum(tf.abs(y-y_))

	l_rate=tf.placeholder(tf.float32) #placeholder for the learning rate

	train_step = tf.train.AdamOptimizer(learning_rate=l_rate,beta1=bet1, beta2=bet2,epsilon=eps).minimize(cost_function_complete)
	init = tf.initialize_all_variables()

	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()

	#launch the graph
	sess = tf.Session()
	sess.run(init)

	# Restore variables from disk.
	if input_net != "":
	  saver.restore(sess, input_net)
	else:
	  print "no input net specified."

	N_training_batch = training_set.get_N()/batch_size #rounds to samllest integer

	cost_training_data = [0]*N_epochs #cost function after each training epoch
	out_mod_validation_data = [0]*N_epochs # output of net, when inputting x_mod of validation data, saved after each epoch
	error_mod_validation_data = [0]*N_epochs # error between x_mod_desired and x_mod_outputted
	total_error_mod_validation_data = [0]*N_epochs #total error between x_mod_des and x_mod_outputted

	reduced = 0 #number of times where the learning rate was reduced
	for i in range(0,N_epochs):
		print "train with range: "+ str(train_range[i%len(train_range)])
		if train_range[i%len(train_range)]==-1: #train with full training set
			for j in range(0,N_training_batch):
				batch_xs, batch_ys, epochs_completed = training_set.next_batch(batch_size)#always gives the modified x's and y's. If one does not want to modifie them the function has to be set to identity 
				sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, dropout_placeholder_1: dropout_keep_prob[0], dropout_placeholder_2: dropout_keep_prob[1],
			                                        dropout_placeholder_3: dropout_keep_prob[2], dropout_placeholder_4: dropout_keep_prob[3],dropout_placeholder_5: dropout_keep_prob[4],
			                                        l_rate : learning_rat}) #makes it possible to change the initial learning rate
		if train_range[i%len(train_range)]!=-1: #train only with training data in specific range
			print "train only with range: "+ str(train_range[i%len(train_range)])
			for j in range(0,training_set.get_Nr(train_range[i%len(train_range)])/batch_size):
				batch_xs, batch_ys, epochs_completed = training_set.next_batch_r(batch_size,train_range[i%len(train_range)])#always gives the modified x's and y's. If one does not want to modifie them the function has to be set to identity 
				sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, dropout_placeholder_1: dropout_keep_prob[0], dropout_placeholder_2: dropout_keep_prob[1],
			                                        dropout_placeholder_3: dropout_keep_prob[2], dropout_placeholder_4: dropout_keep_prob[3],dropout_placeholder_5: dropout_keep_prob[4],
			                                        l_rate : learning_rat*1.0}) #when training with one specific range reduce learning rate by 0.05

		cost_training_data[i] = sess.run(cost_function_complete, feed_dict={
			x: training_set.get_x_mod(), y_: training_set.get_y_mod(), dropout_placeholder_1: 1.0, dropout_placeholder_2: 1.0, dropout_placeholder_3: 1.0,dropout_placeholder_4: 1.0,dropout_placeholder_5: 1.0})
		out_mod_validation_data[i] = sess.run(y, feed_dict={
			x: validation_set.get_x_mod(), dropout_placeholder_1: 1.0, dropout_placeholder_2: 1.0, dropout_placeholder_3: 1.0,dropout_placeholder_4: 1.0,dropout_placeholder_5: 1.0})
		error_mod_validation_data[i] = sess.run(error , feed_dict={
			x: validation_set.get_x_mod(),y_: validation_set.get_y_mod(), dropout_placeholder_1: 1.0, dropout_placeholder_2: 1.0, dropout_placeholder_3: 1.0,dropout_placeholder_4: 1.0,dropout_placeholder_5: 1.0})
		total_error_mod_validation_data[i]=sess.run(total_error , feed_dict={
			x: validation_set.get_x_mod(),y_: validation_set.get_y_mod(), dropout_placeholder_1: 1.0, dropout_placeholder_2: 1.0, dropout_placeholder_3: 1.0,dropout_placeholder_4: 1.0,dropout_placeholder_5: 1.0})
			

		print "epochs completed: "+str(i)
		print "cost training data: "+ str(cost_training_data[i])
		print "total error mod validation data: "+str(total_error_mod_validation_data[i])
                print "output for x_mod validation as input"
                print out_mod_validation_data[i]

		"""#TODO: user this again if we do another hyperparameter scan
		if math.isnan(cost_training_data[i]):#sometimes in extreme situations cost will be nan, then exit
			fr=open("./scan_result_outputs/errors_on_validation_data_"+str(array_id),"a")
			fr.write(str(array_id)+" "+ h_1_str[:-1]+ " nan "+"\n")
			fr.close()
			exit()
		if i>=4:# cancel training if error stays the same for 4 epochs!
			if total_error_mod_validation_data[i]==total_error_mod_validation_data[i-1] and total_error_mod_validation_data[i]==total_error_mod_validation_data[i-2] and total_error_mod_validation_data[i]== total_error_mod_validation_data[i-3] and total_error_mod_validation_data[i]== total_error_mod_validation_data[i-4]:
				fr=open("./scan_result_outputs/errors_on_validation_data_"+str(array_id),"a")
				fr.write(str(array_id)+" "+ h_1_str[:-1]+ " last_4_stagnate_at "+str(total_error_mod_validation_data[i])+" at_epoch "+str(i)+"\n")
				fr.close()
				exit()
		"""	

		if i % 10 == 0 and i>=20: #check every 10th epoch for saturation and if the learning rate will be decreased.
			z=0.0
			n=0.0
			mean_x=0.0
			mean_y=0.0
			for j in range(0,10):
				mean_y=mean_y+total_error_mod_validation_data[i-j]
				print mean_x
				print mean_y
				mean_x=mean_x+i-j
			mean_x=mean_x/10.0
			mean_y=mean_y/10.0

			for j in range(0,10):
				z=z+(total_error_mod_validation_data[i-j]-mean_y)*(i-j-mean_x)
				n=n+(i-j-mean_x)**2.0
			m=z/n
			print "m "+ str(m) #slope of a line which is fitted trough the last 10 points

			if m>-5.0 :# one has to set this threhold according to the transformations one does on the data.
				learning_rat=learning_rat*0.5
				reduced=reduced+1
				print "learning rate reduced by a factor of 0.5"

		print "------------------------------------------------------------------"

	print "learning rate was reduced "+ str(reduced)+ " times by a factor of 0.5"


	# Save the variables to disk.
	if output_net!="":
	  save_path = saver.save(sess, output_net)
	  print("Model saved in file: %s" % save_path)
	else:
	  print "net was not saved to disk!"


	
	out_validation_data = [0]*N_epochs # output of net, when inputting x_mod of validation data and making the normalization backwards, saved after each epoch
	error_validation_data = [0.0]*N_epochs
	total_error_validation_data = np.array([0.0]*N_epochs)
	rel_error_validation_data = [0.0]*N_epochs
	total_rel_error_validation_data = np.array([0.0]*N_epochs)
	classification_validation_data = np.array([0.0]*N_epochs)

	#make the transformation on the outputs backwards
	for i in range(0,N_epochs):
	  out_validation_data[i] = np.ndarray(shape=(validation_set.get_N(),1))
	  for j in range(0,len(out_mod_validation_data[i])):
	    out_validation_data[i][j] = out_mod_validation_data[i][j] #do this, because otherwise we will produce only a reference

	  rd.apply_inverse_function_to_outputs(out_mod_validation_data[i],out_validation_data[i],full_set.get_y_max(),full_set.get_y_min(),full_set.get_y_mean(),full_set.get_y_stddev(),full_set.get_y_mod_max())# second argument will be changed!

	  error_validation_data[i] = np.subtract(out_validation_data[i],validation_set.get_y())
          total_error_validation_data[i] = np.sum(np.absolute(error_validation_data[i]))

	  rel_error_validation_data[i] = np.absolute(error_validation_data[i])/validation_set.get_y()

          total_rel_error_validation_data[i] = np.sum(np.absolute(error_validation_data[i])/validation_set.get_y())

	  classification_validation_data[i] =  np.sum((np.sign(-np.absolute(error_validation_data[i])+classification)+1.0)*0.5)


	print "output (for x_mod_validation as input)"
	print out_mod_validation_data[-1]
	print "desired output y_mod_validation"
	print validation_set.get_y_mod()
	print "total error between them"
	print total_error_mod_validation_data
	print "--------------------------------------------"
	print "--------------------------------------------"
	print "output (for x_validation as input)"
	print out_validation_data[-1]
	print "desired output y_validation"
	print validation_set.get_y()
	print "total error validation data"
	print total_error_validation_data
	print "total rel error validation data"
	print total_rel_error_validation_data
	print "classification validation data"
	print classification_validation_data
	print "--------------------------------------------"
	print "total error validation data/N_val data"
	print total_error_validation_data/validation_set.get_N() 
	print "total rel error validation data/N_val data"
	print total_rel_error_validation_data/validation_set.get_N()
	print "classification validation data/ N_val data"
	print classification_validation_data/validation_set.get_N()
	print "--------------------------------------------"
	print "--------------------------------------------"

	#now do everything for the y ranges seperately
	error_validation_data_ranges = [None]*N_epochs # die kann ich nicht zu kompletten numpy arrays machen, weil die elemente der liste unterscheidliche dimension haben werden!
	total_error_validation_data_ranges = np.ndarray(shape=(N_epochs,len(y_ranges)))
	rel_error_validation_data_ranges = [None]*N_epochs
	total_rel_error_validation_data_ranges = np.ndarray(shape=(N_epochs,len(y_ranges)))
	classification_validation_data_ranges = np.ndarray(shape=(N_epochs,len(y_ranges)))

	for m in range(0,N_epochs):
		error_validation_data_ranges[m] = [np.ndarray(shape=(validation_set.get_Nr(u),N_out)) for u in range(0,len(y_ranges))]
		rel_error_validation_data_ranges[m] = [np.ndarray(shape=(validation_set.get_Nr(u),N_out)) for u in range(0,len(y_ranges))]
		k=[0]*len(y_ranges)
		for i in range(0,validation_set.get_N()):
			for j in range(0,len(y_ranges)):
				if y_ranges[j][0]<validation_set.get_y()[i][0] and validation_set.get_y()[i][0]<=y_ranges[j][1]:
					error_validation_data_ranges[m][j][k[j]][0] = error_validation_data[m][i][0]
					rel_error_validation_data_ranges[m][j][k[j]][0] = rel_error_validation_data[m][i][0]
					k[j]=k[j]+1	

	for m in range(0,N_epochs):
		for j in range(0,len(y_ranges)):
			total_error_validation_data_ranges[m][j] = np.sum(np.absolute(error_validation_data_ranges[m][j]))
			total_rel_error_validation_data_ranges[m][j] = np.sum(np.absolute(rel_error_validation_data_ranges[m][j]))
			classification_validation_data_ranges[m][j] = np.sum((np.sign(-np.absolute(error_validation_data_ranges[m][j])+classification)+1.0)*0.5)

	
	print "total_error_validation_data_ranges"
	print total_error_validation_data_ranges
	print "total_rel_error_validation_data_ranges"
	print total_rel_error_validation_data_ranges
	print "classification_validation_data_ranges in +- "+str(classification)
	print classification_validation_data_ranges
	print "--------------------------------------------"
	print "total_error_validation_data_ranges/ number in range"
	print [total_error_validation_data_ranges[i]/(validation_set.get_Nr_tot()*1.0) for i in range(0,N_epochs)]
	print "total_rel_error_validation_data_ranges/ number in range"
	print [total_rel_error_validation_data_ranges[i]/(validation_set.get_Nr_tot()*1.0) for i in range(0,N_epochs)]
	print "classification_validation_data_ranges/ number in range in +- "+str(classification)
	print [classification_validation_data_ranges[i]/(validation_set.get_Nr_tot()*1.0) for i in range(0,N_epochs)]
	print "--------------------------------------------"
	print "--------------------------------------------"


#append last 10 errors on the validation data to errors_on_validation_data
"""fr=open(outputfolder+"/subtletly_scan_result_outputs","a")

last_10_val_errors=""
for i in range(1,11):
  last_10_val_errors=last_10_val_errors+" "+str(total_error_validation_data[-i])
fr.write(str(array_id_1)+" "+ line_1_str[:-1]+ " "+last_10_val_errors+"\n")

for j in xrange(0,len(y_ranges)):
  last_10_val_errors_range=""
  for i in range(1,11):
    last_10_val_errors_range=last_10_val_errors_range+" "+str(total_error_validation_data_ranges[-i][j])
  fr.write(last_10_val_errors_range+"\n")
  fr.write("-------------------------------------------"+"\n")
fr.close()
"""



# # # # # #
#plotting #
# # # # # #

#plot the number of points in the ranges in a histogram
ranges_hist = array('f',[0]*(len(y_ranges)+1))
ranges_hist[0] = train_ys_only_in_range[0]
for i in range(0,len(y_ranges)):
	ranges_hist[i+1] = y_ranges[i][1]

ranges_hist[-1]=ranges_hist[-1]+0.001

c = TCanvas("c","c",1000,700)			
hist = TH1D('histogram',"",len(y_ranges),ranges_hist)

for i in range(0,len(full_set.get_Nr_tot())):
	hist.SetBinContent(i+1,full_set.get_Nr(i)) 
hist.GetXaxis().SetTitle("#chi^{2}")
hist.GetYaxis().SetTitle("Number of points")
hist.SetLineWidth(2)
ROOT.gPad.Update()
ROOT.gStyle.SetOptStat(0)# Do not print the stat box
hist.Draw()
c.SaveAs(outputfolder+'/number_points_in_ranges'+str(array_id)+'.pdf')



#Draw same histogram with a lot more bins
c = TCanvas("c","c",1000,700)	
c.SetLogy(1)	
hist = TH1D('hist',"",100,train_ys_only_in_range[0],train_ys_only_in_range[1]+0.001)
fill_hist(hist,full_set.get_y().reshape((1,full_set.get_N()))[0])
hist.GetXaxis().SetTitle(r"#chi^{2}")
hist.GetYaxis().SetTitle("Number of points")
hist.SetLineWidth(2)
ROOT.gPad.Update()
ROOT.gStyle.SetOptStat(0)# Do not print the stat box
hist.Draw()
c.SaveAs(outputfolder+'/number_points_in_ranges_more_bins'+str(array_id)+'.pdf')




#Draw the same histogram as before, but only with mathplotlib
mpl.rcParams.update({'font.size': 45,'font.family': 'sans-serif'})
mpl.rcParams['xtick.major.size'] = 20
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.size'] = 20
mpl.rcParams['xtick.minor.width'] = 2

mpl.rcParams['ytick.major.size'] = 40
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 20
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['axes.linewidth'] = 2

fig3=plt.figure(figsize=(20, 15 ))

from matplotlib.ticker import MultipleLocator


plt.hist(full_set.get_y().reshape((1,full_set.get_N()))[0],100, histtype='step',linewidth=2,label=r"All $\chi^2$", range=[train_ys_only_in_range[0],train_ys_only_in_range[1]])
plt.xlabel(r'$\chi^2_{CM,'+str(energy)+'TeV}$', fontsize=50)
plt.ylabel('Number of points', fontsize=48)

for i in range(0,len(y_ranges)):
  plt.plot([y_ranges[i][1],y_ranges[i][1]],[1,100000],linewidth=2,color="black",linestyle="dashed")

minor_locator = MultipleLocator(1)
major_locator = MultipleLocator(5)
ax=plt.gca()
ax.xaxis.set_minor_locator(minor_locator)
ax.xaxis.set_minor_locator(major_locator)

plt.yscale('log', nonposy='clip')
plt.tight_layout()

ax1 = fig3.add_subplot(111)
ax1.text(20, 1000, 'I', fontsize=45)
ax1.text(60, 1000, 'II', fontsize=45)
ax1.text(80, 1000, 'III', fontsize=45)


plt.savefig(outputfolder+'/number_points_in_ranges_more_bins_matplotlib'+str(array_id)+'.pdf')



#cost on training data
plt.figure(1,figsize=(15, 10))
plt.title("Costfunction of (modified) Training-data")
plt.xlabel("Epochs")
plt.ylabel("Cost function")
x_range=[x+1 for x in range(0,N_epochs)]
plt.plot(x_range,cost_training_data,linewidth=2)
plt.savefig(outputfolder+"/cost_on_training_data"+str(array_id)+".pdf")



#total error on validation data
plt.figure(2,figsize=(15, 10 ))
plt.title("")
plt.xlabel("Epochs")
plt.ylabel("Mean error on validation points")
x_range=[x+1 for x in range(0,N_epochs)]
plt.plot(x_range,total_error_validation_data/(validation_set.get_N()*1.0),linewidth=2 )
#if not math.isnan(error_interpol_lin):
#	plt.plot([1,N_epochs],[error_interpol_lin,error_interpol_lin],color="grey")
if not math.isnan(total_error_validation_data_nearest_neighbour):
	plt.plot([1,N_epochs],[total_error_validation_data_nearest_neighbour/(validation_set.get_N()*1.0),total_error_validation_data_nearest_neighbour/(validation_set.get_N()*1.0)],linewidth=2,color="black")
blue_patch = mpatches.Patch(color='blue', label='Neural Net')
#grey_patch = mpatches.Patch(color='grey', label='LinearNDInterpolator')
black_patch = mpatches.Patch(color='black', label='NearestNDInterpolator')

#plt.legend(handles=[black_patch,blue_patch],loc='upper center', bbox_to_anchor=(0.5, 1.10),ncol=2, fancybox=True)
plt.legend(handles=[black_patch,blue_patch],loc='upper center', bbox_to_anchor=(0.5, 1.10),ncol=2)


#axes = plt.gca()
#axes.set_ylim([0,error_validation_data[0]])

plt.savefig(outputfolder+"/total_error_on_val_data_"+str(array_id)+".pdf")




#total error on validation data (with y_max=30)
plt.figure(3,figsize=(15, 10 ))
plt.title("")
plt.xlabel("Epochs")
plt.ylabel("Mean error on validation points")
x_range=[x+1 for x in range(0,N_epochs)]
plt.plot(x_range,total_error_validation_data/(validation_set.get_N()*1.0) ,linewidth=2)
#if not math.isnan(error_interpol_lin):
#	plt.plot([1,N_epochs],[error_interpol_lin,error_interpol_lin],color="grey")
if not math.isnan(total_error_validation_data_nearest_neighbour):
	plt.plot([1,N_epochs],[total_error_validation_data_nearest_neighbour/(validation_set.get_N()*1.0),total_error_validation_data_nearest_neighbour/(validation_set.get_N()*1.0)],linewidth=2,color="black")
blue_patch = mpatches.Patch(color='blue', label='Neural Net')
#grey_patch = mpatches.Patch(color='grey', label='LinearNDInterpolator')
black_patch = mpatches.Patch(color='black', label='NearestNDInterpolator')

plt.legend(handles=[black_patch,blue_patch],loc='upper center', bbox_to_anchor=(0.5, 1.10),ncol=2)

axes = plt.gca()
axes.set_ylim([0,30])

plt.savefig(outputfolder+"/total_error_on_val_data_ylim_"+str(array_id)+".pdf")


#total error on validation data (with y_max=8)
plt.figure(4,figsize=(15, 10 ))
plt.title("")
plt.xlabel("Epochs")
plt.ylabel("Mean error on validation points")
x_range=[x+1 for x in range(0,N_epochs)]
plt.plot(x_range,total_error_validation_data/(validation_set.get_N()*1.0),linewidth=2 )
#if not math.isnan(error_interpol_lin):
#	plt.plot([1,N_epochs],[error_interpol_lin,error_interpol_lin],color="grey")
if not math.isnan(total_error_validation_data_nearest_neighbour):
	plt.plot([1,N_epochs],[total_error_validation_data_nearest_neighbour/(validation_set.get_N()*1.0),total_error_validation_data_nearest_neighbour/(validation_set.get_N()*1.0)],linewidth=2,color="black")
blue_patch = mpatches.Patch(color='blue', label='Neural Net')
#grey_patch = mpatches.Patch(color='grey', label='LinearNDInterpolator')
black_patch = mpatches.Patch(color='black', label='NearestNDInterpolator')

plt.legend(handles=[black_patch,blue_patch],loc='upper center', bbox_to_anchor=(0.5, 1.10),ncol=2)

axes = plt.gca()
axes.set_ylim([0,8])

plt.savefig(outputfolder+"/total_error_on_val_data_ylim1_"+str(array_id)+".pdf")



#total error on validation data for ranges
plt.figure(5,figsize=(15, 10 ))
ax = plt.subplot(111)
plt.title("")
plt.xlabel("Epochs")
plt.ylabel("Mean error on validation points")
x_range=[x+1 for x in range(0,N_epochs)]

names=[]
plots=[None]*len(y_ranges)
for i in range(0,len(y_ranges)):
	names.append(str(y_ranges[i][0])+" < $\chi^2 \leq $ "+str(y_ranges[i][1]))

dashes=[[4,2,1,2],[1,1],[100000,1],[4,2],[10, 5]]
markerstyles=["x",".","v","*","s","+","|","_","^","p"]

for i in range(0,len(y_ranges)):
	line,=ax.plot(x_range,total_error_validation_data_ranges[:,i]/validation_set.get_Nr(i),color="blue",label=r""+names[i],marker=markerstyles[i],linewidth=2,markersize=10)
	#line.set_dashes(dashes[i])
#if not math.isnan(error_interpol_lin):
#	plt.plot([1,N_epochs],[error_interpol_lin,error_interpol_lin],color="grey")
for i in range(0,len(y_ranges)):
	if not math.isnan(total_error_validation_data_nearest_neighbour_ranges[i]):
		plots[i], = ax.plot(range(1,N_epochs+1,10),[total_error_validation_data_nearest_neighbour_ranges[i]/validation_set.get_Nr(i) for k in range(1,N_epochs+1,10)],linewidth=2,color="black",label=names[i],marker=markerstyles[i],markersize=10)
		#plots[i].set_dashes(dashes[i])

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.74, box.height])
plt.legend(handles=[black_patch,blue_patch]+[p for p in plots],loc='center left', bbox_to_anchor=(1.0, 0.5),fontsize="small")

plt.savefig(outputfolder+"/total_error_on_val_data_components_"+str(array_id)+".pdf")





#total error on validation data for ranges (y_lim=10)
plt.figure(6,figsize=(15, 10 ))
ax = plt.subplot(111)
plt.title("")
plt.xlabel("Epochs")
plt.ylabel("Mean error on validation points")
x_range=[x+1 for x in range(0,N_epochs)]


names=[]
plots=[None]*len(y_ranges)
for i in range(0,len(y_ranges)):
	names.append(str(y_ranges[i][0])+" < $\chi^2 \leq$ "+str(y_ranges[i][1]))

for i in range(0,len(y_ranges)):
	line,=ax.plot(x_range,total_error_validation_data_ranges[:,i]/validation_set.get_Nr(i),color="blue",label=r""+names[i],marker=markerstyles[i],linewidth=2,markersize=10)
	#line.set_dashes(dashes[i])
#if not math.isnan(error_interpol_lin):
#	plt.plot([1,N_epochs],[error_interpol_lin,error_interpol_lin],color="grey")
for i in range(0,len(y_ranges)):
	if not math.isnan(total_error_validation_data_nearest_neighbour_ranges[i]):
		plots[i], = ax.plot(range(1,N_epochs+1,10),[total_error_validation_data_nearest_neighbour_ranges[i]/validation_set.get_Nr(i) for k in range(1,N_epochs+1,10)],linewidth=2,markersize=10,color="black",label=names[i],marker=markerstyles[i])
		#plots[i].set_dashes(dashes[i])

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
plt.legend(handles=[black_patch,blue_patch]+[p for p in plots],loc='center left', bbox_to_anchor=(1.0, 0.5),fontsize="small")

axes = plt.gca()
axes.set_ylim([0,10])

plt.savefig(outputfolder+"/total_error_on_val_data_components_y_lim_"+str(array_id)+".pdf")



#total error on validation data for ranges (log scale)
plt.figure(7,figsize=(15, 10 ))
ax = plt.subplot(111)
plt.title("")
plt.xlabel("Epochs")
plt.ylabel("Mean error on validation points")
x_range=[x+1 for x in range(0,N_epochs)]


names=[]
plots=[None]*len(y_ranges)
for i in range(0,len(y_ranges)):
	names.append(str(y_ranges[i][0])+" < $\chi^2 \leq $ "+str(y_ranges[i][1]))

for i in range(0,len(y_ranges)):
	line,=ax.plot(x_range,total_error_validation_data_ranges[:,i]/validation_set.get_Nr(i),color="blue",label=r""+names[i],marker=markerstyles[i],linewidth=2,markersize=10)
	#line.set_dashes(dashes[i])
#if not math.isnan(error_interpol_lin):
#	plt.plot([1,N_epochs],[error_interpol_lin,error_interpol_lin],color="grey")
for i in range(0,len(y_ranges)):
	if not math.isnan(total_error_validation_data_nearest_neighbour_ranges[i]):
		plots[i], = ax.plot(range(1,N_epochs+1,10),[total_error_validation_data_nearest_neighbour_ranges[i]/validation_set.get_Nr(i) for k in range(1,N_epochs+1,10)],linewidth=2,markersize=10,color="black",label=names[i],marker=markerstyles[i])
		#plots[i].set_dashes(dashes[i])

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
plt.legend(handles=[black_patch,blue_patch]+[p for p in plots],loc='center left', bbox_to_anchor=(1.0, 0.5),fontsize="small")

axes = plt.gca()
axes.set_yscale('log')

plt.savefig(outputfolder+"/total_error_on_val_data_components_y_log_scale_"+str(array_id)+".pdf")










#total rel error on validation data
plt.figure(8,figsize=(15, 10 ))
plt.title("")
plt.xlabel("Epochs")
plt.ylabel("Total rel error on validation data/ Number validation points")
x_range=[x+1 for x in range(0,N_epochs)]
plt.plot(x_range,total_rel_error_validation_data/(validation_set.get_N()*1.0),linewidth=2)
#if not math.isnan(error_interpol_lin):
#	plt.plot([1,N_epochs],[error_interpol_lin,error_interpol_lin],color="grey")
if not math.isnan(total_error_validation_data_nearest_neighbour):
	plt.plot([1,N_epochs],[total_rel_error_validation_data_nearest_neighbour/(validation_set.get_N()*1.0),total_rel_error_validation_data_nearest_neighbour/(validation_set.get_N()*1.0)],linewidth=2,color="black")
             
plt.legend(handles=[black_patch,blue_patch],loc='upper center', bbox_to_anchor=(0.5, 1.10),ncol=2)
#axes = plt.gca()
#axes.set_ylim([0,error_validation_data[0]])
plt.savefig(outputfolder+"/total_rel_error_on_val_data_"+str(array_id)+".pdf")



#total rel error on validation data (y_lim 0.06)
plt.figure(9,figsize=(15, 10 ))
plt.title("")
plt.xlabel("Epochs")
plt.ylabel("Total rel error on validation data/ Number validation points")
x_range=[x+1 for x in range(0,N_epochs)]
plt.plot(x_range,total_rel_error_validation_data/(validation_set.get_N()*1.0),linewidth=2)
#if not math.isnan(error_interpol_lin):
#	plt.plot([1,N_epochs],[error_interpol_lin,error_interpol_lin],color="grey")
if not math.isnan(total_error_validation_data_nearest_neighbour):
	plt.plot([1,N_epochs],[total_rel_error_validation_data_nearest_neighbour/(validation_set.get_N()*1.0),total_rel_error_validation_data_nearest_neighbour/(validation_set.get_N()*1.0)],linewidth=2,color="black")
             
plt.legend(handles=[black_patch,blue_patch],loc='upper center', bbox_to_anchor=(0.5, 1.10),ncol=2)
axes = plt.gca()
axes.set_ylim([0,0.06])
plt.savefig(outputfolder+"/total_rel_error_on_val_data_y_lim_"+str(array_id)+".pdf")




#total rel error on validation data for ranges
plt.figure(10,figsize=(15, 10 ))
ax = plt.subplot(111)
plt.title("")
plt.xlabel("Epochs")
plt.ylabel("Total rel error on validation data / Number validation points")

for i in range(0,len(y_ranges)):
	line,=ax.plot(x_range,total_rel_error_validation_data_ranges[:,i]/validation_set.get_Nr(i),color="blue",label=names[i],marker=markerstyles[i],linewidth=2,markersize=10)
	#line.set_dashes(dashes[i])
#if not math.isnan(error_interpol_lin):
#	plt.plot([1,N_epochs],[error_interpol_lin,error_interpol_lin],color="grey")
for i in range(0,len(y_ranges)):
	if not math.isnan(total_rel_error_validation_data_nearest_neighbour_ranges[i]):
		plots[i], = ax.plot(range(1,N_epochs+1,10),[total_rel_error_validation_data_nearest_neighbour_ranges[i]/validation_set.get_Nr(i) for k in range(1,N_epochs+1,10)],linewidth=2,markersize=10,color="black",label=names[i],marker=markerstyles[i])
		#plots[i].set_dashes(dashes[i])

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
plt.legend(handles=[black_patch,blue_patch]+[p for p in plots],loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.savefig(outputfolder+"/total_rel_error_on_val_data_components_"+str(array_id)+".pdf")


#total rel error on validation data for ranges (y_lim=0.06)
plt.figure(11,figsize=(15, 10 ))
ax = plt.subplot(111)
plt.title("")
plt.xlabel("Epochs")
plt.ylabel("Total rel error on validation data / Number validation points")

for i in range(0,len(y_ranges)):
	line,=ax.plot(x_range,total_rel_error_validation_data_ranges[:,i]/validation_set.get_Nr(i),color="blue",label=names[i],marker=markerstyles[i],linewidth=2,markersize=10)
	#line.set_dashes(dashes[i])
#if not math.isnan(error_interpol_lin):
#	plt.plot([1,N_epochs],[error_interpol_lin,error_interpol_lin],color="grey")
for i in range(0,len(y_ranges)):
	if not math.isnan(total_rel_error_validation_data_nearest_neighbour_ranges[i]):
		plots[i], = ax.plot(range(1,N_epochs+1,10),[total_rel_error_validation_data_nearest_neighbour_ranges[i]/validation_set.get_Nr(i) for k in range(1,N_epochs+1,10)],linewidth=2,markersize=10,color="black",label=names[i],marker=markerstyles[i])
		#plots[i].set_dashes(dashes[i])

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
plt.legend(handles=[black_patch,blue_patch]+[p for p in plots],loc='center left', bbox_to_anchor=(1.0, 0.5),fontsize="small")
axes = plt.gca()
axes.set_ylim([0,0.06])
plt.savefig(outputfolder+"/total_rel_error_on_val_data_components_y_lim_"+str(array_id)+".pdf")





#total rel error on validation data for ranges (y_lim=0.06)
plt.figure(111,figsize=(15, 10 ))
ax = plt.subplot(111)
plt.title("")
plt.xlabel("Epochs")
plt.ylabel("Total rel error on validation data / Number validation points")

for i in range(0,len(y_ranges)):
	line,=ax.plot(x_range,total_rel_error_validation_data_ranges[:,i]/validation_set.get_Nr(i),color="blue",label=names[i],marker=markerstyles[i],linewidth=2,markersize=10)
	#line.set_dashes(dashes[i])
#if not math.isnan(error_interpol_lin):
#	plt.plot([1,N_epochs],[error_interpol_lin,error_interpol_lin],color="grey")
for i in range(0,len(y_ranges)):
	if not math.isnan(total_rel_error_validation_data_nearest_neighbour_ranges[i]):
		plots[i], = ax.plot(range(1,N_epochs+1,10),[total_rel_error_validation_data_nearest_neighbour_ranges[i]/validation_set.get_Nr(i) for k in range(1,N_epochs+1,10)],linewidth=2,markersize=10,color="black",label=names[i],marker=markerstyles[i])
		#plots[i].set_dashes(dashes[i])

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
plt.legend(handles=[black_patch,blue_patch]+[p for p in plots],loc='center left', bbox_to_anchor=(1.0, 0.5),fontsize="small")

axes = plt.gca()
axes.set_yscale('log')

plt.savefig(outputfolder+"/total_rel_error_on_val_data_components_y_log_scale_"+str(array_id)+".pdf")



#classification efficiencies
plt.figure(12,figsize=(15, 10 ))
plt.title("")
plt.xlabel("Epochs")
plt.ylabel("Classified correct within an error of "+ str(classification)+" / Number validation points")
x_range=[x+1 for x in range(0,N_epochs)]
plt.plot(x_range,classification_validation_data/(validation_set.get_N()*1.0),linewidth=2 )
#if not math.isnan(error_interpol_lin):
#	plt.plot([1,N_epochs],[error_interpol_lin,error_interpol_lin],color="grey")

plt.plot([1,N_epochs],[classified_correct_validation_data_nearest_neighbour/(validation_set.get_N()*1.0),classified_correct_validation_data_nearest_neighbour/(validation_set.get_N()*1.0)],linewidth=2,color="black")

plt.legend(handles=[black_patch,blue_patch],loc='upper center', bbox_to_anchor=(0.5, 1.10),ncol=2)
#axes = plt.gca()
#axes.set_ylim([0,error_validation_data[0]])
plt.savefig(outputfolder+"/classification_of_val_data_"+str(array_id)+".pdf")



#classification efficiency on validation data for ranges
plt.figure(13,figsize=(15, 10 ))
ax = plt.subplot(111)
plt.title("")
plt.xlabel("Epochs")
plt.ylabel("Classified correct within an error of "+ str(classification)+" / Number validation points")


for i in range(0,len(y_ranges)):
	line,=ax.plot(x_range,classification_validation_data_ranges[:,i]/validation_set.get_Nr(i),color="blue",label=names[i],marker=markerstyles[i],linewidth=2,markersize=10)#linestyle='None'
	#line.set_dashes(dashes[i])	
#if not math.isnan(error_interpol_lin):
#	plt.plot([1,N_epochs],[error_interpol_lin,error_interpol_lin],color="grey")
for i in range(0,len(y_ranges)):
	plots[i], = ax.plot(range(1,N_epochs+1,10),[classified_correct_validation_data_nearest_neighbour_ranges[i]/validation_set.get_Nr(i) for k in range(1,N_epochs+1,10)],linewidth=2,markersize=10,color="black",label=names[i],marker=markerstyles[i])#linestyle='None'
	#plots[i].set_dashes(dashes[i])

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

plt.legend(handles=[black_patch,blue_patch]+[p for p in plots],loc='center left', bbox_to_anchor=(1.0, 0.5),fontsize="small")

#axes = plt.gca()
#axes.set_ylim([-0.2,1.0])
#plt.ylim(-0.2, plt.ylim()[1])
#axes.set_yscale('log')

plt.savefig(outputfolder+"/classification_of_val_data_components_"+str(array_id)+".pdf")



#classification efficiency on validation data for ranges (log scale)
plt.figure(14,figsize=(15, 10 ))
ax = plt.subplot(111)
plt.title("")
plt.xlabel("Epochs")
plt.ylabel("Classified correct within an error of "+ str(classification)+" / Number validation points")


for i in range(0,len(y_ranges)):
	line,=ax.plot(x_range,classification_validation_data_ranges[:,i]/validation_set.get_Nr(i),color="blue",label=names[i],marker=markerstyles[i],linewidth=2,markersize=10)#linestyle='None'
	#line.set_dashes(dashes[i])	
#if not math.isnan(error_interpol_lin):
#	plt.plot([1,N_epochs],[error_interpol_lin,error_interpol_lin],color="grey")
for i in range(0,len(y_ranges)):
	plots[i], = ax.plot(range(1,N_epochs+1,10),[classified_correct_validation_data_nearest_neighbour_ranges[i]/validation_set.get_Nr(i) for k in range(1,N_epochs+1,10)],linewidth=2,markersize=10,color="black",label=names[i],marker=markerstyles[i])#linestyle='None'
	#plots[i].set_dashes(dashes[i])

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

plt.legend(handles=[black_patch,blue_patch]+[p for p in plots],loc='center left', bbox_to_anchor=(1.0, 0.5), fancybox=True, shadow=True,fontsize="small")

axes = plt.gca()
#axes.set_ylim([-0.2,1.0])
#plt.ylim(-0.2, plt.ylim()[1])
axes.set_yscale('log')

plt.savefig(outputfolder+"/classification_of_val_data_components_log_scale_"+str(array_id)+".pdf")




#error on validation data after training hist zoomed nearest comparison
x_zoom=10 
canvas_z=TCanvas("canvas_z","canvas_z",1000,700)			
hist=TH1D('hist',"",100,-x_zoom*5,x_zoom*5)
fill_hist(hist,error_validation_data[-1].reshape((1,validation_set.get_N()))[0]) 
hist.GetXaxis().SetTitle(r"outputted #chi^{2}-desired #chi^{2}") #hist.GetXaxis().SetTitle("outputted \chi^2-desired \chi^2")
hist.GetYaxis().SetTitle("Number of points")
hist.SetLineWidth(2)
hist.Draw()
hist1=TH1D('hist1',"",100,-x_zoom*5,x_zoom*5)
fill_hist(hist1,error_validation_data_nearest_neighbour.reshape((1,validation_set.get_N()))[0]) 
hist1.GetXaxis().SetTitle(r"outputted #chi^{2}-desired #chi^{2}") #hist.GetXaxis().SetTitle("outputted \chi^2-desired \chi^2")
hist1.SetLineColorAlpha(1, 1.0)
hist1.SetLineWidth(2)
hist1.Draw("same")

leg = TLegend(0.1,0.7,0.4,0.9) #x1,y1,x2,y2 
leg.SetHeader("Interpolation methods")
leg.AddEntry(hist1,"NearestNDInterpolator","l")
leg.AddEntry(hist,"Neural net","l")
leg.Draw("same")
ROOT.gPad.Update()
ROOT.gStyle.SetOptStat(0)# Do not print the stat box
canvas_z.Update()
canvas_z.SaveAs(outputfolder+'/error_on_val_data_hist_zoom'+str(array_id)+'.pdf') # TODO: Wenn ich das als pdf speichere verschwindet das achsen label!?


#error on validation data after training hist zoomed log nearest comparison
canvas_z_log=TCanvas("canvas_z_log","canvas_z_log",1000,700)
canvas_z_log.SetLogy(1)			
hist=TH1D('hist',"",100,-x_zoom*10,x_zoom*10)
fill_hist(hist,error_validation_data[-1].reshape((1,validation_set.get_N()))[0]) 
hist.GetXaxis().SetTitle(r"outputted #chi^{2}-desired #chi^{2}")
hist.GetYaxis().SetTitle("Number of points")
hist.SetLineWidth(2)
hist.Draw()
hist1=TH1D('hist1',"",100,-x_zoom*10,x_zoom*10)
fill_hist(hist1,error_validation_data_nearest_neighbour.reshape((1,validation_set.get_N()))[0]) 
hist1.GetXaxis().SetTitle(r"outputted #chi^{2}-desired #chi^{2}")
hist1.SetLineColorAlpha(1, 1.0)
hist1.SetLineWidth(2)
hist1.Draw("same")

leg = TLegend(0.1,0.7,0.4,0.9)
leg.SetHeader("Interpolation methods")
leg.AddEntry(hist1,"NearestNDInterpolator","l")
leg.AddEntry(hist,"Neural net","l")
leg.Draw("same")
canvas_z_log.Update()
canvas_z_log.SaveAs(outputfolder+'/error_on_val_data_hist_zoom_log_'+str(array_id)+'.pdf')



#error on validation data after training hist zoomed log nearest comparison (another zoom)
canvas_z_log=TCanvas("canvas_z_log","canvas_z_log",1000,700)
canvas_z_log.SetLogy(1)			
hist=TH1D('hist',"",100,-x_zoom,x_zoom)
fill_hist(hist,error_validation_data[-1].reshape((1,validation_set.get_N()))[0]) 
hist.GetXaxis().SetTitle(r"outputted #chi^{2}-desired #chi^{2}")
hist.GetYaxis().SetTitle("Number of points")
hist.SetLineWidth(2)
hist.Draw()
hist1=TH1D('hist1',"",100,-x_zoom,x_zoom)
fill_hist(hist1,error_validation_data_nearest_neighbour.reshape((1,validation_set.get_N()))[0]) 
hist1.GetXaxis().SetTitle(r"outputted #chi^{2}-desired #chi^{2}")
hist1.SetLineColorAlpha(1, 1.0)
hist1.SetLineWidth(2)
hist1.Draw("same")

leg = TLegend(0.1,0.7,0.4,0.9)
leg.SetHeader("Interpolation methods")
leg.AddEntry(hist1,"NearestNDInterpolator","l")
leg.AddEntry(hist,"Neural net","l")
leg.Draw("same")
canvas_z_log.Update()
canvas_z_log.SaveAs(outputfolder+'/error_on_val_data_hist_zoom_log_2_'+str(array_id)+'.pdf')



#error histograms above only in matplotlib
fig = plt.figure(figsize=(20, 15 ))

plt.hist(error_validation_data_nearest_neighbour.reshape((1,validation_set.get_N()))[0],39, histtype='step',linewidth=2,label="NearestNDInterpolator", range=[-50, 50],color="black",linestyle="dashed")
plt.hist(error_validation_data[-1].reshape((1,validation_set.get_N()))[0],49, histtype='step',linewidth=2,label="Neural Network", range=[-50, 50],color="blue")


plt.legend(loc='upper right',prop={'size':36})
plt.xlabel(r'$\chi^2_{SN,'+str(energy)+'TeV}-\chi^2_{CM,'+str(energy)+'TeV}$', fontsize=50)
plt.ylabel('Number of points',fontsize=48)

minor_locator = MultipleLocator(1)
major_locator = MultipleLocator(5)
ax=plt.gca()
ax.xaxis.set_minor_locator(minor_locator)
ax.xaxis.set_minor_locator(major_locator)

plt.yscale('log', nonposy='clip')
plt.tight_layout()
plt.savefig(outputfolder+'/error_on_val_data_hist_zoom_log_2_matplotlib'+str(array_id)+'.pdf')




#error on validation data after training hist zoomed with log scale neural nets (compontents separately)
canvas_z_log_neural_net=TCanvas("canvas_z_log_neural_net","canvas_z_log_neural_net",1000,700)			
canvas_z_log_neural_net.SetLogy(1)
hist=TH1D('hist',"",100,-x_zoom,x_zoom)
fill_hist(hist,error_validation_data[-1].reshape((1,validation_set.get_N()))[0]) 
hist.GetXaxis().SetTitle(r"outputted #chi^{2}-desired #chi^{2}")
hist.GetYaxis().SetTitle("Number of points")
hist.SetMinimum(1.0)
hist.SetLineWidth(2)
hist.Draw()

leg = TLegend(0.1,0.65,0.38,0.9) #x1,y1,x2,y2
leg.SetHeader("ranges")
leg.AddEntry(hist,"all #chi^{2}","l")

#plot ranges
histograms=[]
for i in range(0,len(y_ranges)):
	histograms.append(TH1D('hist1',"Errors on val data after last training epoch zoom log Neural net",100,-x_zoom,x_zoom))
	histograms[i].SetLineColorAlpha(i+1, 1.0)
	fill_hist(histograms[i],error_validation_data_ranges[-1][i].reshape((1,validation_set.get_Nr(i)))[0]) 
	histograms[i].GetXaxis().SetTitle(r"outputted #chi^{2}-desired #chi^{2}")
	histograms[i].SetMinimum(1.0)
	histograms[i].SetLineWidth(2)
	histograms[i].Draw("same")
	leg.AddEntry(histograms[i],str(y_ranges[i][0])+" < #chi^{2} #leq "+str(y_ranges[i][1]),"l")

leg.Draw("same")


leg1 = TLegend(0.7,0.7,1.0,0.9)
leg1.SetHeader("caracteristics")
leg1.AddEntry(hist,"#mu ="+str(round(np.mean(error_validation_data[-1],axis=0)[0],1))+", #sigma = "+str(round(np.std(error_validation_data[-1],axis=0)[0],1)),"l")


#legend for the stddev and mu of the distributions
for i in range(0,len(y_ranges)):
	leg1.AddEntry(histograms[i],"#mu ="+str(round(np.mean(error_validation_data_ranges[-1][i],axis=0)[0],1))+", #sigma = "+str(round(np.std(error_validation_data_ranges[-1][i],axis=0)[0],1)),"l")
leg1.Draw("same")


ROOT.gPad.Update() #for deleting statbox
ROOT.gStyle.SetOptStat(0)
canvas_z_log_neural_net.Update()

canvas_z_log_neural_net.SaveAs(outputfolder+'/error_on_val_data_hist_zoom_log_neural_net_components_'+str(array_id)+'.pdf')







#error histograms above only in matplotlib
fig = plt.figure(figsize=(20, 15 ))
plt.hist(error_validation_data[-1].reshape((1,validation_set.get_N()))[0],39, histtype='step',linewidth=2,label=r"All $\chi^2$", range=[-10, 10])


for i in range(0,len(y_ranges)):
  plt.hist(error_validation_data_ranges[-1][i].reshape((1,validation_set.get_Nr(i)))[0],49, histtype='step',linewidth=2,label=str(y_ranges[i][0]) + "< $\chi^2 \leq$ " + str(y_ranges[i][1]), range=[-10, 10])
plt.legend(loc='upper right',prop={'size':36})
plt.xlabel(r'$\chi^2_{SN,'+str(energy)+'TeV}-\chi^2_{CM,'+str(energy)+'TeV}$', fontsize = 50)
plt.ylabel('Number of points', fontsize = 48)

minor_locator = MultipleLocator(1)
major_locator = MultipleLocator(5)
ax=plt.gca()
ax.xaxis.set_minor_locator(minor_locator)
ax.xaxis.set_minor_locator(major_locator)

plt.yscale('log', nonposy='clip')
plt.tight_layout()
plt.savefig(outputfolder+'/error_on_val_data_hist_zoom_log_neural_net_components_matplotlib'+str(array_id)+'.pdf')

exit()





#error on validation data after training hist zoomed with log scale nearest neighbour (compontents seperately)
canvas_z_log_nearest_neighbour=TCanvas("canvas_z_log_nearest_neighbour","canvas_z_log_nearest_neighbour",1000,700)			
canvas_z_log_nearest_neighbour.SetLogy(1)
hist=TH1D('hist',"",100,-x_zoom,x_zoom)
fill_hist(hist,error_validation_data_nearest_neighbour.reshape((1,validation_set.get_N()))[0]) 
hist.GetXaxis().SetTitle(r"outputted #chi^{2}-desired #chi^{2}")
hist.GetYaxis().SetTitle("Number of points")
hist.SetMinimum(1.0)
hist.SetLineWidth(2)
hist.Draw()

leg = TLegend(0.1,0.65,0.38,0.9) #x1,y1,x2,y2
leg.SetHeader("ranges")
leg.AddEntry(hist,"all #chi^{2}","l")

#plot ranges
histograms=[]
for i in range(0,len(y_ranges)):
	histograms.append(TH1D('hist1',"",100,-x_zoom,x_zoom))
	histograms[i].SetLineColorAlpha(i+1, 1.0)
	fill_hist(histograms[i],error_validation_data_nearest_neighbour_ranges[i].reshape((1,validation_set.get_Nr(i)))[0]) 
	histograms[i].GetXaxis().SetTitle(r"outputted #chi^{2}-desired #chi^{2}")
	histograms[i].SetMinimum(1.0)
	histograms[i].SetLineWidth(2)
	histograms[i].Draw("same")
	leg.AddEntry(histograms[i],str(y_ranges[i][0])+" < #chi^{2} #leq  "+str(y_ranges[i][1]),"l")

leg.Draw("same")


leg1 = TLegend(0.7,0.7,1.0,0.9)
leg1.SetHeader("caracteristics")
leg1.AddEntry(hist,"#mu ="+str(round(np.mean(error_validation_data_nearest_neighbour,axis=0)[0],1))+", #sigma = "+str(round(np.std(error_validation_data_nearest_neighbour,axis=0)[0],1)),"l")


#legend for the stddev and mu of the distributions
for i in range(0,len(y_ranges)):
	leg1.AddEntry(histograms[i],"#mu ="+str(round(np.mean(error_validation_data_nearest_neighbour_ranges[i],axis=0)[0],1))+", #sigma = "+str(round(np.std(error_validation_data_nearest_neighbour_ranges[i],axis=0)[0],1)),"l")
leg1.Draw("same")


ROOT.gPad.Update() #for deleting statbox
ROOT.gStyle.SetOptStat(0)
canvas_z_log_nearest_neighbour.Update()

canvas_z_log_nearest_neighbour.SaveAs(outputfolder+'/error_on_val_data_hist_zoom_log_nearest_neighbour_components_'+str(array_id)+'.pdf')








#2d histogram chi^2 error and chi^2
plt.figure(15,figsize=(15, 10 ))
plt.title("")
plt.xlabel(r"outputted $\chi^2$-desired $\chi^2$")
plt.ylabel(r"desired $\chi^2$")
hist=plt.hist2d(error_validation_data[-1].reshape((1,validation_set.get_N()))[0], validation_set.get_y().reshape((1,validation_set.get_N()))[0],cmin=1, bins=[50,50],range=np.array([[-(train_ys_only_in_range[1]-train_ys_only_in_range[0]),(train_ys_only_in_range[1]-train_ys_only_in_range[0])],[train_ys_only_in_range[0],train_ys_only_in_range[1]]]))

plt.colorbar()
plt.savefig(outputfolder+"/error_vs_chi2_"+str(array_id)+".pdf")


#same histo2d only as 3d plot
c2 = TCanvas("c2","c2",1000,700);
c2.SetLogz(1)
#hist_error_vs_chi= TH2D("hist_error_vs_chi","chi^2 error vs chi^2",10,-cut_output_max*0.1,cut_output_max*0.1,50,50,cut_output_max+10)# xbins, xmin, xmax, ybins, ymin, ymax + +10 to get consitency with matplotlib plot
hist_error_vs_chi= TH2D("hist_error_vs_chi","",15,-(train_ys_only_in_range[1]-train_ys_only_in_range[0])*0.3,(train_ys_only_in_range[1]-train_ys_only_in_range[0])*0.3,25,20,train_ys_only_in_range[1]+10)
hist_error_vs_chi.GetXaxis().SetTitle(r"outputted #chi^{2}-desired #chi^{2}")
hist_error_vs_chi.GetYaxis().SetTitle(r"desired #chi^{2}")
hist_error_vs_chi.GetZaxis().SetTitle("Number of points")
for i in range(0,validation_set.get_N()):
	hist_error_vs_chi.Fill(error_validation_data[-1][i],validation_set.get_y()[i][0])
hist_error_vs_chi.GetXaxis().SetTitleOffset(1.6)
hist_error_vs_chi.GetYaxis().SetTitleOffset(1.6)
hist_error_vs_chi.Draw("LEGO")
c2.SaveAs(outputfolder+"/error_vs_chi2_3d_"+str(array_id)+".pdf")





#profile plots#################################

#for M1
plt.figure(16,figsize=(15, 10 ))
plt.title("M1 and rel error on chi^2")
plt.xlabel("M1")
plt.ylabel("rel error on chi^2 (in percent)")
#print validation_set.get_x()[:,0]
#print rel_error_validation[-1]
#print (rel_error_validation[-1].reshape((1,validation_set.get_N()))*100)[0]
plt.scatter(validation_set.get_x()[:,0],(rel_error_validation_data[-1].reshape((1,validation_set.get_N()))*100)[0])# -1 because we want this after the last training epoch

plt.savefig(outputfolder+"/profile_plot_for_M1_"+str(array_id)+".png")

#for M2
plt.figure(17,figsize=(15, 10 ))
plt.title("M2 and rel error on chi^2")
plt.xlabel("M2")
plt.ylabel(r"rel error on $\chi^2$ (in percent)")
plt.scatter(validation_set.get_x()[:,1],(rel_error_validation_data[-1].reshape((1,validation_set.get_N()))*100)[0])# -1 because we want this after the last training epoch

plt.savefig(outputfolder+"/profile_plot_for_M2_"+str(array_id)+".png")


#for M3
plt.figure(18,figsize=(15, 10 ))
plt.title(r"M3 and rel error on $\chi^2$")
plt.xlabel("M3")
plt.ylabel(r"rel error on $\chi^2$ (in percent)")

plt.scatter(validation_set.get_x()[:,2],(rel_error_validation_data[-1].reshape((1,validation_set.get_N()))*100)[0])# -1 because we want this after the last training epoch

plt.savefig(outputfolder+"/profile_plot_for_M3_"+str(array_id)+".png")

#for msq12
plt.figure(19,figsize=(15, 10 ))
plt.title(r"msq12 and rel error on $\chi^2$")
plt.xlabel("msq12")
plt.ylabel(r"rel error on $\chi^2$ (in percent)")

plt.scatter(validation_set.get_x()[:,3],(rel_error_validation_data[-1].reshape((1,validation_set.get_N()))*100)[0])# -1 because we want this after the last training epoch

plt.savefig(outputfolder+"/profile_plot_for_msq12_"+str(array_id)+".png")


#for msq3
plt.figure(20,figsize=(15, 10 ))
plt.title(r"msq3 and rel error on $\chi^2$")
plt.xlabel("msq3")
plt.ylabel(r"rel error on $\chi^2$ (in percent)")

plt.scatter(validation_set.get_x()[:,4],(rel_error_validation_data[-1].reshape((1,validation_set.get_N()))*100)[0])# -1 because we want this after the last training epoch

plt.savefig(outputfolder+"/profile_plot_for_msq3_"+str(array_id)+".png")


#for msl12
plt.figure(21,figsize=(15, 10 ))
plt.title(r"msl12 and rel error on $\chi^2$")
plt.xlabel("msl12")
plt.ylabel(r"rel error on $\chi^2$ (in percent)")

plt.scatter(validation_set.get_x()[:,5],(rel_error_validation_data[-1].reshape((1,validation_set.get_N()))*100)[0])# -1 because we want this after the last training epoch

plt.savefig(outputfolder+"/profile_plot_for_msl12_"+str(array_id)+".png")


#for msl3
plt.figure(22,figsize=(15, 10 ))
plt.title(r"msl3 and rel error on $\chi^2$")
plt.xlabel("msl3")
plt.ylabel(r"rel error on $\chi^2$ (in percent)")

plt.scatter(validation_set.get_x()[:,6],(rel_error_validation_data[-1].reshape((1,validation_set.get_N()))*100)[0])# -1 because we want this after the last training epoch

plt.savefig(outputfolder+"/profile_plot_for_msl3_"+str(array_id)+".png")


#for m_A
plt.figure(23,figsize=(15, 10 ))
plt.title(r"m_A and rel error on $\chi^2$")
plt.xlabel("m_A")
plt.ylabel(r"rel error on $\chi^2$ (in percent)")

plt.scatter(validation_set.get_x()[:,7],(rel_error_validation_data[-1].reshape((1,validation_set.get_N()))*100)[0])# -1 because we want this after the last training epoch

plt.savefig(outputfolder+"/profile_plot_for_mA_"+str(array_id)+".png")

#for A_0
plt.figure(24,figsize=(15, 10 ))
plt.title(r"A_0 and rel error on $\chi^2$")
plt.xlabel("A_0")
plt.ylabel(r"rel error on $\chi^2$ (in percent)")

plt.scatter(validation_set.get_x()[:,8],(rel_error_validation_data[-1].reshape((1,validation_set.get_N()))*100)[0])# -1 because we want this after the last training epoch

plt.savefig(outputfolder+"/profile_plot_for_A0_"+str(array_id)+".png")

#for mu
plt.figure(25,figsize=(15, 10 ))
plt.title(r"mu and rel error on $\chi^2$")
plt.xlabel("mu")
plt.ylabel(r"rel error on $\chi^2$ (in percent)")

plt.scatter(validation_set.get_x()[:,9],(rel_error_validation_data[-1].reshape((1,validation_set.get_N()))*100)[0])# -1 because we want this after the last training epoch

plt.savefig(outputfolder+"/profile_plot_for_mu_"+str(array_id)+".png")


#for tanbeta
plt.figure(26,figsize=(15, 10 ))
plt.title(r"tan(beta) and rel error on $\chi^2$")
plt.xlabel("tan(beta)")
plt.ylabel(r"rel error on $\chi^2$ (in percent)")

plt.scatter(validation_set.get_x()[:,10],(rel_error_validation_data[-1].reshape((1,validation_set.get_N()))*100)[0])# -1 because we want this after the last training epoch

plt.savefig(outputfolder+"/profile_plot_for_tanbeta_"+str(array_id)+".png")



plt.figure(27,figsize=(15, 10 ))
plt.title("")
plt.xlabel("M1")
plt.ylabel(r"correct $\chi^2$")
cm=plt.cm.get_cmap('RdYlBu')

sc = plt.scatter(validation_set.get_x()[:,0], validation_set.get_y().reshape((1,validation_set.get_N()))[0], c=(rel_error_validation_data[-1].reshape((1,validation_set.get_N()))*100)[0], s=35, cmap=cm)
plt.colorbar(sc)
plt.savefig(outputfolder+"/profile1_plot_for_M1_"+str(array_id)+".png")


plt.figure(28,figsize=(15, 10 ))
plt.title("")
plt.xlabel("M2")
plt.ylabel(r"correct $\chi^2$")
cm=plt.cm.get_cmap('RdYlBu')

sc = plt.scatter(validation_set.get_x()[:,1], validation_set.get_y().reshape((1,validation_set.get_N()))[0], c=(rel_error_validation_data[-1].reshape((1,validation_set.get_N()))*100)[0], s=35, cmap=cm)
plt.colorbar(sc)
plt.savefig(outputfolder+"/profile1_plot_for_M2_"+str(array_id)+".png")


plt.figure(29,figsize=(15, 10 ))
plt.title("")
plt.xlabel("M3")
plt.ylabel(r"correct $\chi^2$")
cm=plt.cm.get_cmap('RdYlBu')

sc = plt.scatter(validation_set.get_x()[:,2], validation_set.get_y().reshape((1,validation_set.get_N()))[0], c=(rel_error_validation_data[-1].reshape((1,validation_set.get_N()))*100)[0], s=35, cmap=cm)
plt.colorbar(sc)
plt.savefig(outputfolder+"/profile1_plot_for_M3_"+str(array_id)+".png")


plt.figure(30,figsize=(15, 10 ))
plt.title("")
plt.xlabel("msq12")
plt.ylabel(r"correct $\chi^2$")
cm=plt.cm.get_cmap('RdYlBu')

sc = plt.scatter(validation_set.get_x()[:,3], validation_set.get_y().reshape((1,validation_set.get_N()))[0], c=(rel_error_validation_data[-1].reshape((1,validation_set.get_N()))*100)[0], s=35, cmap=cm)
plt.colorbar(sc)
plt.savefig(outputfolder+"/profile1_plot_for_msq12_"+str(array_id)+".png")


plt.figure(31,figsize=(15, 10 ))
plt.title("")
plt.xlabel("msq3")
plt.ylabel(r"correct $\chi^2$")
cm=plt.cm.get_cmap('RdYlBu')

sc = plt.scatter(validation_set.get_x()[:,4], validation_set.get_y().reshape((1,validation_set.get_N()))[0], c=(rel_error_validation_data[-1].reshape((1,validation_set.get_N()))*100)[0], s=35, cmap=cm)
plt.colorbar(sc)
plt.savefig(outputfolder+"/profile1_plot_for_msq3_"+str(array_id)+".png")


plt.figure(32,figsize=(15, 10 ))
plt.title("")
plt.xlabel("msl12")
plt.ylabel(r"correct $\chi^2$")
cm=plt.cm.get_cmap('RdYlBu')

sc = plt.scatter(validation_set.get_x()[:,5], validation_set.get_y().reshape((1,validation_set.get_N()))[0], c=(rel_error_validation_data[-1].reshape((1,validation_set.get_N()))*100)[0], s=35, cmap=cm)
plt.colorbar(sc)
plt.savefig(outputfolder+"/profile1_plot_for_msl12_"+str(array_id)+".png")


plt.figure(33,figsize=(15, 10 ))
plt.title("")
plt.xlabel("msl3")
plt.ylabel(r"correct $\chi^2$")
cm=plt.cm.get_cmap('RdYlBu')

sc = plt.scatter(validation_set.get_x()[:,6], validation_set.get_y().reshape((1,validation_set.get_N()))[0], c=(rel_error_validation_data[-1].reshape((1,validation_set.get_N()))*100)[0], s=35, cmap=cm)
plt.colorbar(sc)
plt.savefig(outputfolder+"/profile1_plot_for_msl3_"+str(array_id)+".png")


plt.figure(34,figsize=(15, 10 ))
plt.title("")
plt.xlabel("M_A")
plt.ylabel(r"correct $\chi^2$")
cm=plt.cm.get_cmap('RdYlBu')

sc = plt.scatter(validation_set.get_x()[:,7], validation_set.get_y().reshape((1,validation_set.get_N()))[0], c=(rel_error_validation_data[-1].reshape((1,validation_set.get_N()))*100)[0], s=35, cmap=cm)
plt.colorbar(sc)
plt.savefig(outputfolder+"/profile1_plot_for_mA_"+str(array_id)+".png")


plt.figure(35,figsize=(15, 10 ))
plt.title("")
plt.xlabel("A0")
plt.ylabel(r"correct $\chi^2$")
cm=plt.cm.get_cmap('RdYlBu')

sc = plt.scatter(validation_set.get_x()[:,8], validation_set.get_y().reshape((1,validation_set.get_N()))[0], c=(rel_error_validation_data[-1].reshape((1,validation_set.get_N()))*100)[0], s=35, cmap=cm)
plt.colorbar(sc)
plt.savefig(outputfolder+"/profile1_plot_for_A0_"+str(array_id)+".png")


plt.figure(36,figsize=(15, 10 ))
plt.title("")
plt.xlabel("mu")
plt.ylabel(r"correct $\chi^2$")
cm=plt.cm.get_cmap('RdYlBu')

sc = plt.scatter(validation_set.get_x()[:,9], validation_set.get_y().reshape((1,validation_set.get_N()))[0], c=(rel_error_validation_data[-1].reshape((1,validation_set.get_N()))*100)[0], s=35, cmap=cm)
plt.colorbar(sc)
plt.savefig(outputfolder+"/profile1_plot_for_mu_"+str(array_id)+".png")


plt.figure(37,figsize=(15, 10 ))
plt.title("")
plt.xlabel("tan(beta)")
plt.ylabel(r"correct $\chi^2$")
cm=plt.cm.get_cmap('RdYlBu')

sc = plt.scatter(validation_set.get_x()[:,10], validation_set.get_y().reshape((1,validation_set.get_N()))[0], c=(rel_error_validation_data[-1].reshape((1,validation_set.get_N()))*100)[0], s=35, cmap=cm)
plt.colorbar(sc)
plt.savefig(outputfolder+"/profile1_plot_for_tanbeta_"+str(array_id)+".png")











