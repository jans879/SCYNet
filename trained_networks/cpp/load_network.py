import sys
import math
import numpy as np

# since the code was written in tensorflow1 and tensorflow2 is now state of the art I've to manually disable tensorflow2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



def apply_function_to_inputs(x,x_mod,x_max,x_min,x_mean,x_stddev,x_mod_max):
   #z_score normalization
   for i in range(0,len(x)):
      for j in range(0,len(x[i])):
        x_mod[i][j]=(x[i][j]-x_mean[j])/x_stddev[j]
   
   for i in range(0,len(x)):
      for j in range(0,len(x[i])):
        x_mod[i][j]=x_mod[i][j]/x_mod_max[j]


def apply_inverse_function_to_outputs(y_mod,y,y_max,y_min,y_mean,y_stddev,y_mod_max):
   #z score normalization
   for i in range(0,len(y)):
      for j in range(0,len(y[i])):
        y[i][j]=y_mod[i][j]*y_mod_max[j]

   for i in range(0,len(y)):
      for j in range(0,len(y[i])):
        y[i][j]=y[i][j]*y_stddev[j]+y_mean[j]



N_in=11
N_out=1

N=[N_in,300,300,300,300,N_out]
activation_functions=["tanh","tanh","tanh","tanh","tanh"]


#create variables for weights and biases
x_in = tf.placeholder(tf.float32,shape=(1,N[0])) # net will be evaluated always for one batch
a=[x_in] #outputs of all layers
w=[]
b=[]
for i in range(0,len(N)-1):
  w.append(tf.Variable(tf.random_normal([N[i], N[i+1]],mean=0.0,stddev=1.0/math.sqrt(N[i]*1.0)),name="w_"+str(i))) #weights[0]=W2
  b.append(tf.Variable(tf.random_normal([N[i+1]]),name="b_"+str(i))) # biases[0]=b2
  a.append(tf.tanh(tf.matmul(a[i], w[i]) + b[i])) 

y = a[len(N)-1] #output of the net




###################################
#restore the network 
###################################


# Add ops to save and restore all the variables.
saver = tf.train.Saver()

o = np.ndarray(shape=(1,1))#backtransformed output of net
o_mod=np.ndarray(shape=(1,1))#output of the net

# Launch the model, use the saver to restore variables from disk, and
sess = tf.Session()
# Restore variables from disk. (only for 13 TeV
saver.restore(sess, "./network/net_13TeV.ckpt")
  






x_max=None
x_min=None
y_max=None
y_min=None
x_mean=None
y_mean=None
x_stddev=None
y_stddev=None
x_mod_max=None
y_mod_max=None


# this is for 13 TeV
x_max=np.array([ 3999.49492236 , 3999.78823978 , 3996.61020348 , 4991.45682079 , 4936.19202124,
  2999.83531432 , 3999.96091867 , 3999.93524406 , 4998.67595591,  4999.45034776,
    59.99075022])
x_min=np.array([ -3.99877535e+03,   1.00043706e+02 , -3.99515436e+03 ,  3.00038370e+02,
   1.46599005e+02 ,  1.00042998e+02 ,  1.00678566e+02 ,  7.27788257e-01,
  -4.99992884e+03,  -4.99821200e+03 ,  1.02348006e+00])
y_max=np.array([ 100.])
y_min=np.array([ 50.91384164])
x_mean=np.array([   16.89192785 , 1247.71638499 ,  676.96779325 , 1979.76186282 , 2242.18640787,
  1393.52862555 , 1759.81711878,  1694.8915135  ,   89.76303812  , -71.12378301,
    23.48578218])
y_mean=np.array([ 62.7])
x_stddev=np.array([ 1562.00381696 ,  917.38722271,  1502.88543666 , 1025.90463054 ,  966.65400264,
   650.55863351 ,  858.09362194  , 931.8065378 ,  2022.99833176 , 1955.17512945,
    13.54399907])
y_stddev=np.array([ 18.69027328])
x_mod_max=np.array([ 2.57084345 , 2.9999021  , 3.108768   , 2.93564808,  2.78693887 , 2.46911901,
  2.61060535 , 2.47373638 , 2.51591501,  2.59341174 , 2.69528725])
y_mod_max=np.array([ 1.99569046])



x = np.ones(shape=(1,11))
x_mod=np.ones(shape=(1,11))

o = np.ndarray(shape=(1,1))#backtransformed output of net
o_mod=np.ndarray(shape=(1,1))#output of the net
