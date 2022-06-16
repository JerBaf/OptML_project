import torch
import random
import numpy as np
import torch.nn as nn
import os
import sys
import getopt
# Custom helpers
from model import *


def main(argv):
    # Parse arguments
    try:
        opts, args = getopt.getopt(argv,"h",["seeds=","output_dir="])
    except getopt.GetoptError:
        print(argv)
        print("Error in inputs, please be sure to provide at least the path to the control file.")
        sys.exit(2)
    seeds = [0,1024,2022]
    output_dir = ""
    for opt, arg in opts:
        if opt == "-h":
            print("\n")
            print("Help section:\n")
            print("This script will instantiate basic model with standard VGG architecture.")
            print("for CIFAR10, CIFAR100, MNIST and FashionMNIST based on the given seeds.\n")
            print("Parameter:\n")
            print("-output_dir: (String) Basename given to the output folder. Please select a name which is")
            print("\t not already in the working directory.\n")
            print("Keyword Arguments:\n")
            print("-seeds: (List) List of seeds for random generation in format [seed1,seed2,...]. (default=[0,1024,2022])\n")
            print("Outputs:\n")
            print("The script will output all the combinations of model and seeds in the output folder.")
            sys.exit(0)
        elif opt == "--seeds":
            seeds_list = arg
            seeds = list(map(lambda s: int(s), seeds_list[1:-1].split(",")))
        elif opt == "--output_dir":
            output_dir = arg
    # Inputs Assertion
    if len(seeds) <= 0:
        print("The provided seeds are unvalid or missing, please provide seeds in format [seed1,seed2,...]")
        sys.exit(1)
    if os.path.isdir(output_dir):
        print(f"The folder {output_dir} already exists, please delete it or provide another name")
        sys.exit(1)
    try:
        os.mkdir(output_dir)
    except:
        print(f"Impossible to create output folder {output_dir}. Please be sure of your permissions.")
        sys.exit(1)
    ### Model instantiation Pipeline
    model_types = {"CIFAR_10":(3,10),"CIFAR_100":(3,100),"MNIST":(1,10),"Fashion_MNIST":(1,10)}
    for s in seeds:
        for name, (in_channel,class_nb) in model_types.items():
            torch.manual_seed(s)
            random.seed(s)
            np.random.seed(s)
            ### Model
            model = VGG(in_channel,class_nb)
            torch.save({"model_state_dict":model.state_dict()},output_dir+"/"+name+"_"+str(s)+".pth")
    print("Creation of initial models ended up with success.")


if __name__ == "__main__":
    main(sys.argv[1:])