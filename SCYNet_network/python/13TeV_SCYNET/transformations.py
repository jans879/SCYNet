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
