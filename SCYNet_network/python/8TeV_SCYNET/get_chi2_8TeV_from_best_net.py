import sys
import math
import numpy as np
import tensorflow as tf
import transformations as trafo

"""
obtain 8TeV LHC chi^2.
Arguments when starting the python script
sys.argv[1]=M_1, sys.argv[2]=M_2, sys.argv[3]=M_3, sys.argv[4]=msq12, sys.argv[5]=msq3
sys.argv[6]=msl12, sys.argv[7]=msl3, sys.argv[8]=M_A , sys.argv[9]=A_0 ,sys.argv[10]=mu, sys.argv[11]=tan(beta)

for example a valid call from command line is:
python get_chi2_from_best_net.py -1682.23027931  3465.09031358  1237.59831623  2429.59820408  3450.31039446 2169.17885836  1800.49227919  1271.54318728 -4832.47899659  -610.85511464 33.96309872
output for this parameterpoint:
38.4782770183
all quantities should be given in GeV
"""

try:
	##############
	#input values#
	##############


	x=np.ndarray(shape=(1,11))
	for i in xrange(1,len(sys.argv)):
	  x[0][i-1]=float(sys.argv[i])


	#check if the given parameterpoint (x) lie in the ranges for which the net was trained
	if x[0][0]>4000.0 or x[0][0]<-4000.0:
	  print "Warning M_1 out of valid range. The net was not tested for points in this parameterrange." 
	if x[0][1]>4000.0 or x[0][1]<100.0:
	  print "Warning M_2 out of valid range. The net was not tested for points in this parameterrange." 
	if x[0][2]>4000.0 or x[0][2]<-4000.0 or ( x[0][2]<400.0 and x[0][2]>-400.0):
	  print "Warning M_3 out of valid range. The net was not tested for points in this parameterrange."       
	if x[0][3]>5000.0 or x[0][3]<300.0:
	  print "Warning msq12 out of valid range. The net was not tested for points in this parameterrange." 
	if x[0][4]>5000.0 or x[0][4]<100.0:
	  print "Warning msq3 out of valid range. The net was not tested for points in this parameterrange." 
	if x[0][5]>3000.0 or x[0][5]<100.0:
	  print "Warning msl12 out of valid range. The net was not tested for points in this parameterrange." 
	if x[0][6]>4000.0 or x[0][6]<100.0:
	  print "Warning msl3 out of valid range. The net was not tested for points in this parameterrange." 
	if x[0][7]>4000.0 or x[0][7]<0.0:
	  print "Warning M_A out of valid range. The net was not tested for points in this parameterrange." 
	if x[0][8]>5000.0 or x[0][8]<-5000.0:
	  print "Warning A_0 out of valid range. The net was not tested for points in this parameterrange." 
	if x[0][9]>5000.0 or x[0][9]<-5000.0 or ( x[0][9]<100.0 and x[0][9]>-100.0):
	  print "Warning mu out of valid range. The net was not tested for points in this parameterrange." 
	if x[0][10]>60.0 or x[0][10]<1.0:
	  print "Warning tan(beta) out of valid range. The net was not tested for points in this parameterrange." 


	#for the transformation of input and output values we need:
	x_max=np.array([ 3998.23504782 , 3999.78823978 , 3996.61020348 , 4997.13243765 , 4985.61836211,
	2999.83531432 , 3999.50016348 , 3999.93507767 , 4998.67595591 , 4999.45034776,
	59.9941286 ])
	x_min=np.array([ -3.99907016e+03 ,  1.00035947e+02,  -3.99420860e+03,   3.00038370e+02,
	1.46599005e+02  , 1.00215626e+02 ,  1.07158432e+02 ,  1.14057116e+01,
	-4.99992884e+03 , -4.99782140e+03   ,1.02348006e+00])
	y_max=np.array([ 100.])
	y_min=np.array([ 31.06917746])
	x_mean=np.array([   22.80628857,  1490.92536241 ,  894.14577569  ,2450.64955504,  2680.83189436,
	1620.77668953 , 2013.16041976 , 2001.50975689 ,  -66.57097025  , -55.78971629,
	26.20602196])
	y_mean=np.array([ 55.95])
	x_stddev=np.array([ 1765.37162051 , 1065.75765934 , 1919.26151219 , 1222.73306978  ,1088.62733005,
	704.31351228 ,  933.20169094 , 1043.19115111,  2358.96682234 , 2257.24183993,
	15.39152788])
	y_stddev=np.array([ 23.00711265])
	x_mod_max=np.array([ 2.27820386,  2.35406507 , 2.54699755 , 2.08261553 , 2.32791592 , 2.15892644,
	2.12852137 , 1.91568469 , 2.1472311  , 2.23956511 , 2.19524058])
	y_mod_max=np.array([ 1.9146253])


	x_mod=np.zeros(shape=(1,11))
	trafo.apply_function_to_inputs(x,x_mod,x_max,x_min,x_mean,x_stddev,x_mod_max)




	########################
	#setting up the network#
	########################

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
	#restore the network and get chi^2#
	###################################


	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()

	o = np.ndarray(shape=(1,1))#backtransformed output of net
	o_mod=np.ndarray(shape=(1,1))#output of the net

	# Launch the model, use the saver to restore variables from disk, and
	with tf.Session() as sess:
	  # Restore variables from disk.
	  saver.restore(sess, "./net_8TeV.ckpt")
	  
	  o_mod=sess.run(y,feed_dict={x_in:x_mod})# modified output

	 
	trafo.apply_inverse_function_to_outputs(o_mod,o,y_max,y_min,y_mean,y_stddev,y_mod_max)
	  
	#print "predicted chi^2"
	print o[0][0]

except:
	print "-1"

