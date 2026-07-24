#x will be always set directly from the cpp file before calling this script

apply_function_to_inputs(x,x_mod,x_max,x_min,x_mean,x_stddev,x_mod_max)

o_mod = sess.run(y,feed_dict={x_in:x_mod})# modified output

apply_inverse_function_to_outputs(o_mod,o,y_max,y_min,y_mean,y_stddev,y_mod_max)

# write the chi2 to the file chi2.txt
# each writing with the option "w" overwrites
with open("chi2.txt", "w") as fa:
    fa.write(str(o[0][0]))
