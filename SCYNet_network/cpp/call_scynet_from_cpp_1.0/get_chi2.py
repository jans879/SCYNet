

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


if energy == 8:
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
elif energy==13:
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



x= np.ones(shape=(1,11))
#read in the parameter point from textfile
fr = open("parameterpoint.txt")
line = fr.readlines()[0].split( )
fr.close()

#fill array
for i in range(0,len(line)):
  x[0][i]=float(line[i])


x_mod=np.ones(shape=(1,11))
apply_function_to_inputs(x,x_mod,x_max,x_min,x_mean,x_stddev,x_mod_max)

o = np.ndarray(shape=(1,1))#backtransformed output of net
o_mod=np.ndarray(shape=(1,1))#output of the net

o_mod = sess.run(y,feed_dict={x_in:x_mod})# modified output

 
apply_inverse_function_to_outputs(o_mod,o,y_max,y_min,y_mean,y_stddev,y_mod_max)
  
# write the chi2 to the file chi2.txt
# each writing with the option "w" overwrites 
fa = open("chi2.txt", "w")
fa.write(str(o[0][0]))
fa.close()


