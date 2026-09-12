
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np

###########################
# Setup Plotting Defaults #
###########################
# For more options see https://matplotlib.org/users/customizing.html

# Line styles
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['lines.antialiased'] = True
mpl.rcParams['lines.dashed_pattern'] = 2.8, 1.5
mpl.rcParams['lines.dashdot_pattern'] = 4.8, 1.5, 0.8, 1.5
mpl.rcParams['lines.dotted_pattern'] = 1.1, 1.1
mpl.rcParams['lines.scale_dashes'] = True

# Default colors
#from cycler import cycler
#mpl.rcParams['axes.prop_cycle'] = cycler('color',['cornflowerblue','forestgreen','maroon','goldenrod','firebrick','mediumorchid'])


# Fonts
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'CMU Serif'
mpl.rcParams['font.sans-serif'] = 'CMU Sans Serif, DejaVu Sans, Bitstream Vera Sans, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif'
mpl.rcParams['text.usetex'] = True

# Axes
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['axes.labelsize'] = 25
mpl.rcParams['axes.labelpad'] = 9.0
                                                  
                                                  
# Tick marks - the essence of life
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.minor.size'] = 2.5
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['xtick.minor.width'] = 0.75
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.minor.size'] = 2.5
mpl.rcParams['ytick.major.width'] = 1.0
mpl.rcParams['ytick.minor.width'] = 0.75
mpl.rcParams['ytick.major.pad'] = 8
mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['ytick.minor.visible'] = True

# Legend
mpl.rcParams['legend.fontsize'] = 22
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.framealpha'] = 0.8
#mpl.rcParams['legend.edgecolor'] = 'black'
mpl.rcParams['legend.fancybox'] = False
mpl.rcParams['legend.borderpad'] = 0.4 # border whitespace
mpl.rcParams['legend.labelspacing'] = 0.5 # the vertical space between the legend entries
mpl.rcParams['legend.handlelength'] = 1.5 # the length of the legend lines
mpl.rcParams['legend.handleheight'] = 0.7 # the height of the legend handle
mpl.rcParams['legend.handletextpad'] = 0.5 # the space between the legend line and legend text
mpl.rcParams['legend.borderaxespad'] = 0.5 # the border between the axes and legend edge
mpl.rcParams['legend.columnspacing'] = 2.0 # column separation


# Figure size
mpl.rcParams['figure.figsize'] = 20, 14

# Save details
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.1

mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20





##############################
# Visualize training progress
##############################

energy = "8"

input_dir = "./network_performance_data/"

output_dir = "./network_performance_plots/"

# Visualize training

total_error_validation_data = np.load(input_dir+'total_error_validation_data_'+energy+'TeV.npy')
mean_total_error_validation_data = np.load(input_dir+'mean_total_error_validation_data_'+energy+'TeV.npy')
total_rel_error_validation_data = np.load(input_dir+'total_rel_error_validation_data_'+energy+'TeV.npy')
classification_validation_data = np.load(input_dir+'classification_validation_data_'+energy+'TeV.npy')

total_error_validation_data_ranges = np.load(input_dir+'total_error_validation_data_ranges_'+energy+'TeV.npy')
mean_total_error_validation_data_ranges = np.load(input_dir+'mean_total_error_validation_data_ranges_'+energy+'TeV.npy')
total_rel_error_validation_data_ranges = np.load(input_dir+'total_rel_error_validation_data_ranges_'+energy+'TeV.npy')
classification_validation_data_ranges = np.load(input_dir+'classification_validation_data_ranges_'+energy+'TeV.npy')


total_error_validation_data_nearest_neighbour = np.load(input_dir+'total_error_validation_data_nearest_neighbour_'+energy+'TeV.npy')
mean_total_error_validation_data_nearest_neighbour = np.load(input_dir+'mean_total_error_validation_data_nearest_neighbour_'+energy+'TeV.npy')
total_rel_error_validation_data_nearest_neighbour = np.load(input_dir+'total_rel_error_validation_data_nearest_neighbour_'+energy+'TeV.npy')
classified_correct_validation_data_nearest_neighbour = np.load(input_dir+'classified_correct_validation_data_nearest_neighbour_'+energy+'TeV.npy')

total_error_validation_data_nearest_neighbour_ranges = np.load(input_dir+'total_error_validation_data_nearest_neighbour_ranges_'+energy+'TeV.npy')
mean_total_error_validation_data_nearest_neighbour_ranges = np.load(input_dir+'mean_total_error_validation_data_nearest_neighbour_ranges_'+energy+'TeV.npy')
total_rel_error_validation_data_nearest_neighbour_ranges = np.load(input_dir+'total_rel_error_validation_data_nearest_neighbour_ranges_'+energy+'TeV.npy')
classified_correct_validation_data_nearest_neighbour_ranges = np.load(input_dir+'classified_correct_validation_data_nearest_neighbour_ranges_'+energy+'TeV.npy')



### Plot
fig, ax = plt.subplots(1, figsize=(10, 8))

plt.plot(mean_total_error_validation_data,color="black",label = r'$0<\chi^2\leq 100$',linewidth=1.5)

range_labels = [r'$0<\chi^2\leq 38$',r'$38<\chi^2\leq 42$',r'$42<\chi^2\leq 70$',r'$70<\chi^2\leq 95$',r'$95<\chi^2\leq 100$']
markers = ['.','o','s','*','^']

for i in range(0,len(mean_total_error_validation_data_ranges.T)):
    plt.plot(mean_total_error_validation_data_ranges.T[i],color="gray",linewidth=0.5,label=range_labels[i],marker = markers[i])

plt.xlabel("Epoch",fontsize=22)
plt.ylabel("Mean Error Validation Data",fontsize=22)

plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.savefig(output_dir+'mean_total_error_validation_data_'+energy+'_TeV.pdf',bbox_inches='tight')


#####################################################
# Write function that plots the data in a histogram
#####################################################


# I call the function from the other script because I don't want to re-load the data here again. I already to that in the other script


def plot_histogram(y,ranges,energy,x_scale,y_scale,x_label,y_label,output_dir):
    fig, ax = plt.subplots(1, figsize=(10, 8))
    plt.hist(y, bins=50, color='skyblue', edgecolor='black',alpha=0.3)
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for i in range(0,len(ranges)):
        plt.axvline(x=ranges[i][0],ls='--',c='black')

    plt.savefig(output_dir+'histogram_data_'+energy+'_TeV.pdf',bbox_inches='tight')
    

"""
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
plt.ylabel("Classified correct within an error of "+ str(xlassification)+" / Number validation points")


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
fig = plt.figure(124,figsize=(15, 10 ))

plt.hist(error_validation_data_nearest_neighbour.reshape((1,validation_set.get_N()))[0],49, histtype='step',linewidth=2,label="NearestNDInterpolator", range=[-50, 50],color="black")
plt.hist(error_validation_data[-1].reshape((1,validation_set.get_N()))[0],49, histtype='step',linewidth=2,label="Neural Network", range=[-50, 50],color="blue")


plt.legend(loc='upper right',prop={'size':26})
plt.xlabel(r'$\chi^2_{SN}-\chi^2_{CM}$', fontsize=28)
plt.ylabel('Number of points')
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
fig = plt.figure(123,figsize=(15, 10 ))
plt.hist(error_validation_data[-1].reshape((1,validation_set.get_N()))[0],49, histtype='step',linewidth=2,label=r"All $\chi^2$", range=[-10, 10])


for i in range(0,len(y_ranges)):
  plt.hist(error_validation_data_ranges[-1][i].reshape((1,validation_set.get_Nr(i)))[0],49, histtype='step',linewidth=2,label=str(y_ranges[i][0]) + "< $\chi^2 \leq$ " + str(y_ranges[i][1]), range=[-10, 10])
plt.legend(loc='upper right',prop={'size':26})
plt.xlabel(r'$\chi^2_{SN}-\chi^2_{CM}$', fontsize=28)
plt.ylabel('Number of points')
plt.yscale('log', nonposy='clip')
plt.tight_layout()
plt.savefig(outputfolder+'/error_on_val_data_hist_zoom_log_neural_net_components_matplotlib'+str(array_id)+'.pdf')







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
"""








