import sys
import math
import numpy as np

# since the code was written in tensorflow1 and tensorflow2 is now state of the art I've to manually disable tensorflow2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import transformations as trafo

"""
obtain 13TeV LHC chi^2.
Arguments when starting the python script (the 11 pMSSM model parameters)
sys.argv[1]=M_1, sys.argv[2]=M_2, sys.argv[3]=M_3, sys.argv[4]=msq12, sys.argv[5]=msq3
sys.argv[6]=msl12, sys.argv[7]=msl3, sys.argv[8]=M_A , sys.argv[9]=A_0 ,sys.argv[10]=mu, sys.argv[11]=tan(beta)

for example a valid call from command line is:
python3 get_chi2_13TeV_from_best_net.py -1682.23027931  3465.09031358  1237.59831623  2429.59820408  3450.31039446 2169.17885836  1800.49227919  1271.54318728 -4832.47899659  -610.85511464 33.96309872
output for this parameterpoint:
89.5022272981
all quantities should be given in GeV
"""

try:
	##############
	#input values#
	##############


	x=np.ndarray(shape=(1,11))
	for i in range(1,len(sys.argv)):
	  x[0][i-1]=float(sys.argv[i])


	#check if the given parameterpoint (x) lies in the ranges for which the net was trained
	if x[0][0]>4000.0 or x[0][0]<-4000.0:
	  print("Warning M_1 out of valid range. The net was not tested for points in this parameterrange.")
	if x[0][1]>4000.0 or x[0][1]<100.0:
	  print("Warning M_2 out of valid range. The net was not tested for points in this parameterrange.")
	if x[0][2]>4000.0 or x[0][2]<-4000.0 or ( x[0][2]<400.0 and x[0][2]>-400.0):
	  print("Warning M_3 out of valid range. The net was not tested for points in this parameterrange.")    
	if x[0][3]>5000.0 or x[0][3]<300.0:
	  print("Warning msq12 out of valid range. The net was not tested for points in this parameterrange.")
	if x[0][4]>5000.0 or x[0][4]<100.0:
	  print("Warning msq3 out of valid range. The net was not tested for points in this parameterrange.")
	if x[0][5]>3000.0 or x[0][5]<100.0:
	  print("Warning msl12 out of valid range. The net was not tested for points in this parameterrange.")
	if x[0][6]>4000.0 or x[0][6]<100.0:
	  print("Warning msl3 out of valid range. The net was not tested for points in this parameterrange.")
	if x[0][7]>4000.0 or x[0][7]<0.0:
	  print("Warning M_A out of valid range. The net was not tested for points in this parameterrange.")
	if x[0][8]>5000.0 or x[0][8]<-5000.0:
	  print("Warning A_0 out of valid range. The net was not tested for points in this parameterrange.")
	if x[0][9]>5000.0 or x[0][9]<-5000.0 or ( x[0][9]<100.0 and x[0][9]>-100.0):
	  print("Warning mu out of valid range. The net was not tested for points in this parameterrange.")
	if x[0][10]>60.0 or x[0][10]<1.0:
	  print("Warning tan(beta) out of valid range. The net was not tested for points in this parameterrange.")


	#for the transformation of input and output values we need:
	x_max=np.array([ 3999.49492236 , 3999.78823978 , 3996.61020348 , 4991.45682079 , 4936.19202124,
  2999.83531432 , 3999.96091867,  3999.93524406 , 4998.67595591 , 4999.45034776,
    59.99075022])
	x_min=np.array([ -3.99877535e+03 ,  1.00043706e+02 , -3.99515436e+03 ,  3.00038370e+02,
   1.46599005e+02 ,  1.00042998e+02 ,  1.00678566e+02  , 7.27788257e-01,
  -4.99992884e+03  ,-4.99821200e+03 ,  1.02348006e+00])
	y_max=np.array([ 100.])
	y_min=np.array([ 50.91384164])
	x_mean=np.array([   16.89192785 , 1247.71638499 ,  676.96779325 , 1979.76186282 , 2242.18640787,
  1393.52862555  ,1759.81711878 , 1694.8915135 ,    89.76303812  , -71.12378301,
    23.48578218])
	y_mean=np.array([ 62.7])
	x_stddev=np.array([ 1562.00381696 ,  917.38722271  ,1502.88543666,  1025.90463054  , 966.65400264,
   650.55863351 ,  858.09362194 ,  931.8065378  , 2022.99833176 , 1955.17512945,
    13.54399907])
	y_stddev=np.array([ 18.69027328])
	x_mod_max=np.array([ 2.57084345 , 2.9999021 ,  3.108768 ,   2.93564808  ,2.78693887 , 2.46911901,
  2.61060535,  2.47373638,  2.51591501 , 2.59341174 , 2.69528725])
	y_mod_max=np.array([ 1.99569046])


	x_mod=np.zeros(shape=(1,11))
	trafo.apply_function_to_inputs(x,x_mod,x_max,x_min,x_mean,x_stddev,x_mod_max)




	########################
	#setting up the network#
	########################

	N_in=11
	N_out=1

    # The network has 4 hidden layers with 300 neurons each. The chosen network parameters were found to be optimal with a hyperparameter scan.
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
	#restore the network and get chi^2#
	###################################


	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()

	o = np.ndarray(shape=(1,1))#backtransformed output of net
	o_mod=np.ndarray(shape=(1,1))#output of the net

	# Launch the model, use the saver to restore variables from disk, and
	with tf.Session() as sess:
	  # Restore variables from disk.
	  saver.restore(sess, "./net_13TeV.ckpt")
	  
	  o_mod=sess.run(y,feed_dict={x_in:x_mod})# modified output

	 
	trafo.apply_inverse_function_to_outputs(o_mod,o,y_max,y_min,y_mean,y_stddev,y_mod_max)
	  
	#print "predicted chi^2"
	print(o[0][0])

except Exception as e:
    import traceback
    traceback.print_exc()

