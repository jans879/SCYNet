
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

plt.savefig(output_dir+'mean_total_error_validation_data_'+energy+'_TeV.png',bbox_inches='tight')


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

    plt.savefig(output_dir+'histogram_data_'+energy+'_TeV.png',bbox_inches='tight')
    


