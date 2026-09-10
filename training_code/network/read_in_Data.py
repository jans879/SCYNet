#################################
# This code was written by Jan Schuette-Engel (janschue@berkeley.de)
# Use this code only with explicit permission.
# Please cite arxiv:1703.01309 if you want to use this code
#################################


import numpy as np
import itertools
import math
from random import randint

#import root_numpy
#from ROOT import TFile, TH2D, TCanvas


def apply_function_to_inputs(x,x_mod,x_max,x_min,x_mean,x_stddev,x_mod_max):# all arguments are references
    #normalize the inputs
    x_mod[:] = (x-x_mean.T)/x_stddev.T # Note that strictly we don't need the transpose because python broadcasts this automatically in the correct shape. The [:] because otherwise pythond does not change the orignal per reference and just creates a new array

    #determine x_mod_max
    x_mod_max[:] = np.max(np.abs(x_mod),axis=0)

    #do the normalization
    x_mod[:] = x_mod/x_mod_max.T


def apply_inverse_function_to_inputs(x_mod,x,x_max,x_min,x_mean,x_stddev,x_mod_max):# all arguments are references
    #re normalize the inputs
    one = x_mod*x_mod_max.T

    #make Z-score backwards
    x[:] = one*x_stddev.T + x_mean.T

    

def apply_function_to_outputs(y,y_mod,y_max,y_min,y_mean,y_stddev,y_mod_max):# all arguments are references
    # The hyperparameter scan was carried out with tanh (except in the last layer I did tanh and linear) and z-score normalization
    #Z-score normalization
    #TODO: implement the modified Z-score normalization!?
    y_mod[:] = (y-y_mean.T)/y_stddev.T

    y_mod_max[:] = np.max(np.abs(y_mod),axis=0)

    y_mod[:] = y_mod/y_mod_max


def apply_inverse_function_to_outputs(y_mod,y,y_max,y_min,y_mean,y_stddev,y_mod_max):# all arguments are references
    #z score normalization
    one = y_mod*y_mod_max.T

    # Z-score backwards
    y[:] = one*y_stddev.T + y_mean.T 


def cut_output(y,cut_output_max,smooth_cut_range,cut_output_min):
    output = y
    if y>cut_output_max-smooth_cut_range and y<=cut_output_max+smooth_cut_range:
        output = -1.0/(4.0*smooth_cut_range)*(y-(cut_output_max+smooth_cut_range))**2+cut_output_max
    elif y>cut_output_max+smooth_cut_range:
        output = cut_output_max

    if y<cut_output_min:
        output = cut_output_min

    return output   


def consider_this_id(array_id,do_not_consider_these_array_ids):
    if do_not_consider_these_array_ids[0][0] == -1:
        return True
    else:
        for j in range(0,len(do_not_consider_these_array_ids)):
            if array_id>=do_not_consider_these_array_ids[j][0] and array_id< do_not_consider_these_array_ids[j][1]:
                return False
    return True

def fill_ranges(N,N_in,N_out,y_ranges,y,x,y_mod,x_mod,yr,xr,yr_mod,xr_mod):
    k = [0]*len(y_ranges)
    for i in range(0,N):
        for j in range(0,len(y_ranges)):
            if y[i][0]>y_ranges[j][0] and y[i][0]<=y_ranges[j][1]:
                yr[j][k[j]][0] = y[i][0]
                yr_mod[j][k[j]][0] = y[i][0]
                for l in range(0,N_in):
                    xr[j][k[j]][l] = x[i][l]
                    xr_mod[j][k[j]][l] = x_mod[i][l]
                k[j]=k[j]+1



class Dataset(object):
  def __init__(self,path,file_name,kind_of_set,cut_output_max,cut_output_min,smooth_cut_range,N_in,N_out,y_ranges,train_ys_only_in_range,extend_data_artificially_in_ranges,do_not_consider_these_array_ids,full_set=None,subset=None):
    
    self._kind_of_set=kind_of_set

    #initialize the data sets
    self._x = None
    self._y = None

    self._y_max =None
    self._y_min = None
    self._y_mean = None
    self._y_stddev = None
    self._y_mod = None #function applied to output values
    self._y_mod_max = None # maximum of y_mod

    self._x_max = None
    self._x_min = None
    self._x_mean = None
    self._x_stddev = None
    self._x_mod = None #function applied to inputs
    self._x_mod_max = None # maximum of x_mod

	#initialize arrays lists for validation and training set indices. Fill them only if kind_of_set= "validation_set" or "training_set"
    self._validation_indices = None
    self._training_indices = None

    # number of input and output nodes of the network
    self._N_in = N_in
    self._N_out = N_out

    self._N = 0 # number of points in set
    self._Nr = np.array([0]*len(y_ranges)) # number of points in the specific ranges # the r stands for ranges


    # y_ranges is for example [[0.0,38.0],[38.0,42.0],[42.0,100.0]]. I put all x and y values in the correct yr and xr arrays
    self._xr = [None]*len(y_ranges)
    self._yr = [None]*len(y_ranges)
    self._xr_mod = [None]*len(y_ranges)# mod means transformed values
    self._yr_mod = [None]*len(y_ranges)

    #Option to extend data artificially.
    self._N_ea = 0 # total number of data points with artificially extended data
    self._Nr_ea = np.array([0]*len(y_ranges)) # artificially extended data in specific target ranges

    self._x_ea = self._x
    self._x_ea_mod = self._x_mod
    self._y_ea = self._y
    self._y_ea_mod = self._y_mod
    self._xr_ea = self._xr
    self._xr_ea_mod = self._xr_mod
    self._yr_ea = self._yr
    self._yr_ea_mod = self._yr_mod


    self._subset = subset # if the data set is training or validation set these are the indices of the full set that correspond to training or validation set. If it's the full set this is just None

    #Full set
    if kind_of_set == "full_set":

        f=open(path+"/"+file_name,'r')
        lines=f.readlines()
        f.close()


        #determine number of points (self._N, self._Nr)
        for i in range(0,len(lines)):
            line = lines[i].split( )
            array_id = int(line[0])
            chi2 = float(line[-1])

            #cut the read out target smoothly at cut_output_max
            chi2 = cut_output(chi2,cut_output_max,smooth_cut_range,cut_output_min)

            #proceed only if the chi2 is in the range given by train_ys_only_in_range
            if chi2 < train_ys_only_in_range[0] or chi2 > train_ys_only_in_range[1]:
                continue

            #check if the array ID should be considered
            if consider_this_id(array_id,do_not_consider_these_array_ids) == False:
                continue

            #count the point
            self._N = self._N+1
            for j in range(0,len(y_ranges)):
                if chi2 > y_ranges[j][0] and chi2 <= y_ranges[j][1]: # if one does not do the cut before the last range needs special treatment, but we cut before
                    self._Nr[j] = self._Nr[j]+1

        #determine the data sets and fill them
        self._x = np.ndarray(shape=(self._N,N_in))
        self._y = np.ndarray(shape=(self._N,N_out))

        k=0
        for i in range(0,len(lines)):
            line = lines[i].split( )
            array_id=int(line[0])
            y = float(line[-1])

            #cut smoothly again:
            y = cut_output(y,cut_output_max,smooth_cut_range,cut_output_min)

            #proceed only if the chi2 is in the range given by train_ys_only_in_range
            if y<train_ys_only_in_range[0] or y> train_ys_only_in_range[1]:
                continue

            #if do_not_consider_these_array_ids != [[-1]] one has to check if the array id is not in the range
            if consider_this_id(array_id,do_not_consider_these_array_ids) == False:
                continue

            for j in range(1,len(line)-1):
                self._x[k][j-1] = float(line[j])
            self._y[k][0] = y
            k = k+1	

        #determine y_min and y_max
        self._y_max = np.max(self._y,axis=0)
        self._y_min = np.min(self._y,axis=0)
        self._y_mean = np.mean(self._y,axis=0)
        self._y_stddev = np.std(self._y,axis=0)
        self._y_mod = np.ndarray(shape=(self._N,N_out)) #function applied to outputs,
        self._y_mod_max = np.zeros(shape=(N_out))

        self._x_max = np.max(self._x,axis=0)
        self._x_min = np.min(self._x,axis=0)
        self._x_mean = np.mean(self._x,axis=0) # x has structure x = [[M1,M2, ...],[M1,M2,...], ... ]. Them x_mean [M1average, M2average, ... (11 entries)]
        self._x_stddev = np.std(self._x,axis=0)
        self._x_mod = np.ndarray(shape=(self._N,N_in)) #function applied to inputs
        self._x_mod_max = np.zeros(shape=(N_in))

        #apply function to inputs and outputs, the function can also be the identity
        apply_function_to_inputs(self._x,self._x_mod,self._x_max,self._x_min,self._x_mean,self._x_stddev,self._x_mod_max)
        apply_function_to_outputs(self._y,self._y_mod,self._y_max,self._y_min,self._y_mean,self._y_stddev,self._y_mod_max)

        #initializing:
        for i in range(0,len(y_ranges)):
            self._xr[i] = np.ndarray(shape=(self._Nr[i],self._N_in))
            self._xr_mod[i] = np.ndarray(shape=(self._Nr[i],self._N_in))
            self._yr[i] = np.ndarray(shape=(self._Nr[i],self._N_out))
            self._yr_mod[i] = np.ndarray(shape=(self._Nr[i],self._N_out))

        #filling the ranges
        fill_ranges(self._N,self._N_in,self._N_out,y_ranges,self._y,self._x,self._y_mod,self._x_mod,self._yr,self._xr,self._yr_mod,self._xr_mod)


    elif kind_of_set=="training_set" or kind_of_set=="validation_set":

        self._N = len(self._subset) #Number of elements of the data set

        self._y = full_set._y[subset]
        self._x = full_set._x[subset]

        self._y_mod = full_set._y_mod[subset]
        self._x_mod = full_set._x_mod[subset]


        # We use the same y_max and x_max ranges in for the full set.
        self._y_max = full_set._y_max
        self._x_max = full_set._x_max
        
        self._y_min = full_set._y_min
        self._x_min = full_set._x_min

        self._y_mean = full_set._y_mean
        self._x_mean = full_set._x_mean

        self._y_stddev = full_set._y_stddev
        self._x_stddev = full_set._x_stddev

        self._y_mod_max = full_set._y_mod_max
        self._x_mod_max = full_set._x_mod_max

        #determine everything for the ranges, count how many points are in the ranges
        for j in range(0,len(y_ranges)):
            self._Nr[j] = np.sum((self._y.T > y_ranges[j][0]) & (self._y.T <= y_ranges[j][1]))
        
        # initialize the xr and yr arrays
        for i in range(0,len(y_ranges)):
            self._xr[i] = np.ndarray(shape=(self._Nr[i],self._N_in))
            self._yr[i] = np.ndarray(shape=(self._Nr[i],self._N_out))
            self._xr_mod[i] = np.ndarray(shape=(self._Nr[i],self._N_in))
            self._yr_mod[i] = np.ndarray(shape=(self._Nr[i],self._N_out))


        fill_ranges(self._N,self._N_in,self._N_out,y_ranges,self._y,self._x,self._y_mod,self._x_mod,self._yr,self._xr,self._yr_mod,self._xr_mod)

        # per default initialize the ea = extended artificially arrays
        # I will just set the following because of consistency reasons
        self._x_ea = self._x
        self._x_ea_mod = self._x_mod
        self._y_ea = self._y
        self._y_ea_mod = self._y_mod

        self._N_ea = self._N
        
        self._xr_ea = self._xr
        self._xr_ea_mod = self._xr_mod
        self._yr_ea = self._yr
        self._yr_ea_mod = self._yr_mod

        self._Nr_ea = self._Nr

    elif self._kind_of_set=="training_set" and extended_artificially_in_ranges != [1]*len(self._Nr): # only in the case of the training set extend the data artificially
    #Comment: if extended_artificially_in_ranges=[1,1,1,1,1], then the 'extended data equals the training data' and no extension is made (doesn't go in here)
    # for this x_mean, y_mean, x_stddev, y_stddev, ... will be the same as before!

        for i in range(0,len(extend_data_artificially_in_ranges)):
            self._Nr_ea[i] = self._Nr[i]*extend_data_artificially_in_ranges[i]
            self._N_ea = self._N_ea + self._Nr_ea[i]

        self._x_ea = np.ndarray(shape=(self._N_ea,self._N_in))
        self._x_ea_mod = np.ndarray(shape=(self._N_ea,self._N_in))
        self._y_ea = np.ndarray(shape=(self._N_ea,self._N_out))
        self._y_ea_mod = np.ndarray(shape=(self._N_ea,self._N_out))

        k = 0
        for i in range(0,self._N):
            for j in range(0,len(extend_data_artificially_in_ranges)):
                if extend_data_artificially_in_ranges[j]!= 1 and (self._y[i][0]>y_ranges[j][0] and self._y[i][0]<=y_ranges[j][1]):
                    for l in range(0,extend_data_artificially_in_ranges[j]):
                        self._y_ea[k][0] = self._y[i][0]# copy all components
                        for m in range(0,self._N_in):
                            self._x_ea[k][m] = self._x[i][m]
                            self._x_ea[k][randint(0,self._N_in-1)] = self._x_ea[k][m]+0.01 # shift one component randomly
                        k=k+1
                elif extend_data_artificially_in_ranges[j]== 1 and ( self._y[i][0]>y_ranges[j][0] and self._y[i][0]<=y_ranges[j][1]):
                    self._y_ea[k][0] = self._y[i][0]
                    for m in range(0,self._N_in):
                        self._x_ea[k][m] = self._x[i][m]
                    k=k+1

        #one has to apply the functions again, because one changes the x value! The y's could be copied in principle, but I apply the function which is equivalent.
        apply_function_to_inputs(self._x_ea,self._x_ea_mod,self._x_max,self._x_min,self._x_mean,self._x_stddev,self._x_mod_max)
        apply_function_to_outputs(self._y_ea,self._y_ea_mod,self._y_max,self._y_min,self._y_mean,self._y_stddev,self._y_mod_max)

        #sort the extended data in ranges:
		#initializing:
        for i in range(0,len(y_ranges)):
            self._xr_ea[i]=np.ndarray(shape=(self._Nr_ea[i],self._N_in))
            self._xr_ea_mod[i]=np.ndarray(shape=(self._Nr_ea[i],self._N_in))
            self._yr_ea[i]=np.ndarray(shape=(self._Nr_ea[i],self._N_out))
            self._yr_ea_mod[i]=np.ndarray(shape=(self._Nr_ea[i],self._N_out))

		#filling the ranges:
        fill_ranges(self._N_ea,self._N_in,self._N_out,y_ranges,self._y_ea,self._x_ea,self._x_ea_mod,self._y_ea_mod,self._yr_ea,self._xr_ea,self._yr_ea_mod,self._xr_ea_mod)


    self._index_in_epoch = 0 #if one has trained 2 mini batches in the epoch already then this is 2*batch_size
    self._epochs_completed = 0

    self._index_in_epoch_r = [0]*len(y_ranges) #if one has trained 2 mini batches in the epoch already then this is 2*batch_size
    self._epochs_completed_r = [0]*len(y_ranges)


  def next_batch(self, batch_size):
      start = self._index_in_epoch
      self._index_in_epoch += batch_size
      if self._index_in_epoch >= self._N_ea:
      # Finished epoch
          self._epochs_completed += 1
          # Shuffle the data
          perm = np.arange(self._N_ea)
          np.random.shuffle(perm)
          self._x_ea = self._x_ea[perm] #shuffle both, actually one would only need to shuffle x_mod and y_mod, but for consistency we shuffle both!
          self._y_ea = self._y_ea[perm]
          self._x_ea_mod = self._x_ea_mod[perm]
          self._y_ea_mod = self._y_ea_mod[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size
          assert batch_size <= self._N_ea #if batch size<= self._N then an exception is thrown!
      end = self._index_in_epoch
      return self._x_ea_mod[start:end], self._y_ea_mod[start:end], self._epochs_completed

  # we need this if we for example want to train only with training data in one range.
  def next_batch_r(self, batch_size,r):
      start = self._index_in_epoch_r[r]
      self._index_in_epoch_r[r] += batch_size
      if self._index_in_epoch_r[r] >= self._Nr_ea[r]:
          # Finished epoch
          self._epochs_completed_r[r] += 1
          # Shuffle the data
          perm = np.arange(self._Nr_ea[r])
          np.random.shuffle(perm)
          self._xr_ea[r] = self._xr_ea[r][perm] #shuffle both, actually one would only need to shuffle x_mod and y_mod, but for consistency we shuffle both!
          self._yr_ea[r] = self._yr_ea[r][perm]
          self._xr_ea_mod[r] = self._xr_ea_mod[r][perm]
          self._yr_ea_mod[r] = self._yr_ea_mod[r][perm]
          # Start next epoch
          start = 0
          self._index_in_epoch_r[r] = batch_size
          assert batch_size <= self._Nr_ea[r] #if batch size<= self._N then an exception is thrown!
      end = self._index_in_epoch_r[r]
      return self._xr_ea_mod[r][start:end], self._yr_ea_mod[r][start:end], self._epochs_completed_r[r]



def read_data_set(path,file_name,kind_of_set,cut_output_max,cut_output_min,smooth_cut_range,N_in,N_out,y_ranges,train_ys_only_in_range,extend_data_artificially_in_ranges,do_not_consider_these_array_ids,full_set=None,subset=None):
    return Dataset(path,file_name,kind_of_set,cut_output_max,cut_output_min,smooth_cut_range,N_in,N_out,y_ranges,train_ys_only_in_range,extend_data_artificially_in_ranges,do_not_consider_these_array_ids,full_set,subset)


