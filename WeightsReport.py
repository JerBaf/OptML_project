import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tqdm.notebook as tqdm
import os
import sys
import getopt
from scipy.fft import fft, ifft


def main(argv):
    # Parse arguments
    try:
        opts, args = getopt.getopt(argv,"h",["seeds=","output_dir=","input_dir=","save","show"])
    except getopt.GetoptError:
        print(argv)
        print("Error in inputs, please be sure to provide at least the path to the control file.")
        sys.exit(2)
    seeds = [0,1024,2022]
    show = False
    save = False
    output_dir = "weights_report"
    input_dir = ""
    for opt, arg in opts:
        if opt == "-h":
            print("\n")
            print("Help section:\n")
            print("This script will generate a report with graph and fourier description")
            print("of the given weights. Please check the github and the paper for extended description.\n")
            print("Parameter:\n")
            print("input_dir: (String) Input folder name where the weights can be found. Pay attention to")
            print("\tto have the weights named cnn_weights.npy and linear_weights.npy in the input folder.\n")
            print("Keyword Arguments:\n")
            print("-output_dir: (String) Basename given to the output folder. Please select a name which is")
            print("\t not already in the working directory. (default=weights_report)")
            print("-seeds: (List) List of seeds for random generation in format [seed1,seed2,...]. (default=[0,1024,2022])")
            print("-save: (Flag) Indicates if the output figures should be saved. (default=False)")
            print("-show: (Flag) Indicates if the figures should be displaed. (default=False)\n")
            print("Outputs:\n")
            print("The script will output all the figures output folder.")
            sys.exit(0)
        elif opt == "--seeds":
            seeds_list = arg
            seeds = list(map(lambda s: int(s), seeds_list[1:-1].split(",")))
        elif opt == "--output_dir":
            output_dir = arg
        elif opt == "--input_dir":
            input_dir = arg
        elif opt == "--save":
            save = True
        elif opt == "--show":
            show = True
    # Inputs Assertion
    if len(seeds) <= 0:
        print("The provided seeds are unvalid or missing, please provide seeds in format [seed1,seed2,...]")
        sys.exit(1)
    if not os.path.isdir(input_dir):
        print("The input folder provided does not exist. Please check the script inputs.")
    if os.path.isdir(output_dir):
        print(f"The folder {output_dir} already exists, please delete it or provide another name")
        sys.exit(1)
    try:
        os.mkdir(output_dir)
    except:
        print(f"Impossible to create output folder {output_dir}. Please be sure of your permissions.")
        sys.exit(1)
    ### Reproducibility
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    ### Weight report pipeline
    weight_report_dict = weights_report(input_dir+"/",output_folder=output_dir,show=show,save=save)
    print("Creation of the weight report ended up with success.")


### Helpers

def weights_report(weights_folder_path,output_folder="weights_report",show=False,save=False):
    """ Generates the weights report based on weight reduction and frequency analysis. """
    ### Load weights
    cnn_weights,linear_weights = load_weights_for_viz(weights_folder_path)
    time_serie_size = cnn_weights.shape[0]
    ### Output Dict
    output_dict = dict()
    ### Plot embedded weights subspace
    for weights in [cnn_weights,linear_weights]:
        embedded_2d, embedded_3d = reduce_weights_dim(weights)
        fig_2d = plot_embedded_2d(embedded_2d,time_serie_size,show)
        fig_3d = plot_embedded_3d(embedded_3d,time_serie_size,show)
        high_freq_ratio, fig_fourier = fourier_features(weights,show)
        base_fig_name = ""
        if "first_cnn" not in output_dict.keys():
            base_fig_name = "first_cnn"
            output_dict["first_cnn"] = [high_freq_ratio,fig_2d,fig_3d,fig_fourier]
        else:
            base_fig_name = "linear"
            output_dict["linear"] = [high_freq_ratio,fig_2d,fig_3d,fig_fourier]
        if save:
            fig_2d.savefig(output_folder+"/"+base_fig_name+"_fig_2d.png")
            fig_3d.savefig(output_folder+"/"+base_fig_name+"_fig_3d.png")
            fig_fourier.savefig(output_folder+"/"+base_fig_name+"_fig_fourier.png")
    return output_dict

def save_weights_for_viz(cnn_weights,linear_weights,basename):
    """ Save weighsts arrays as binary numpy files.  """
    cnn_file = open(basename+"cnn_weights.npy","wb")
    linear_file = open(basename+"linear_weights.npy","wb")
    np.save(cnn_file,cnn_weights)
    np.save(linear_file,linear_weights)
    cnn_file.close()
    linear_file.close()

def load_weights_for_viz(basename):
    """ Load layer weights for further investigation. """
    cnn_file = open(basename+"cnn_weights.npy","rb")
    linear_file = open(basename+"linear_weights.npy","rb")
    cnn_weights = np.load(cnn_file)
    linear_weights = np.load(linear_file)
    cnn_file.close()
    linear_file.close()
    return cnn_weights,linear_weights

def reduce_weights_dim(weights):
    """ Reduce weights dimension first by PCA then by T-SNE. """
    ### PCA dimension reduction
    pca = PCA(n_components=50,svd_solver="auto")
    weights_pca = pca.fit(weights)
    proj_weights = weights_pca.transform(weights)
    ### T-SNE dimension reduction
    weights_embedded_2d = TSNE(n_components=2, learning_rate='auto',
                   init='pca').fit_transform(proj_weights)
    weights_embedded_3d = TSNE(n_components=3, learning_rate='auto',
                   init='pca').fit_transform(proj_weights)
    return weights_embedded_2d, weights_embedded_3d

def plot_embedded_2d(weights_embedded,size,show=False):
    """ Plot weight embedded in 2d. """
    fig, ax = plt.subplots(figsize=(16,9))
    cm = plt.cm.get_cmap('RdYlBu')
    color_range = range(size)
    sc = ax.scatter(weights_embedded[:,0],weights_embedded[:,1], 
                    c=color_range, vmin=0, vmax=size, s=35, cmap=cm)
    plt.colorbar(sc)
    if show:
        plt.show()
    return fig

def plot_embedded_3d(weights_embedded,size,show=False):
    """ Plot weights embedded in 3d. """
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    cm = plt.cm.get_cmap('RdYlBu')
    color_range = range(size)
    sc = ax.scatter(weights_embedded[:,0],weights_embedded[:,1],weights_embedded[:,2], 
                    c=color_range, vmin=0, vmax=size, s=35, cmap=cm)
    plt.colorbar(sc)
    if show:
        plt.show()
    return fig

def fourier_features(weights,high_freq_tr=10,show=False):
    """ Get fast fourier transform of each weight over the different time steps."""
    ### We perform the fft on each weight of the weight vector
    ### then sum their module to have a general approximation of 
    ### the frequency content of the weight vector (and thus it's evolution over time)
    weights_fft = np.abs(fft(weights,axis=0)).sum(axis=-1)
    fig,ax = plt.subplots(figsize=(16,9))
    ax.plot(np.arange(weights_fft.shape[0]//2),np.log(weights_fft)[:weights_fft.shape[0]//2])
    if show:
        plt.show()
    high_freq_ratio = np.sum(weights_fft[high_freq_tr:])/np.sum(weights_fft)
    return high_freq_ratio, fig

if __name__ == "__main__":
    main(sys.argv[1:])

