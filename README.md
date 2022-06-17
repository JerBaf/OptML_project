# Behavior and generalisation capabilities of various optimising techniques for Image Classification using Convolution Neural Networks

This repo contains all the code and figures that we produced during our project.

## Abstract

In this project, we implemented 4 optimizers algorithms: SGD, SGD with momentum, RMSprop and ADAM and investigated the impact of them as well as its learning rate for solving a classification problem on multiple datasets: CIFAR10, CIFAR100, MNIST, FashionMNIST. We implemented the testing on the augmented data by Gaussian, flip and affine transforms in order to demonstrate the generalization capacities of optimizers. The behaviour analysis seem to show that momentum helps in generalization, which is also proven by the best accuracy reached by MSGD. 

## Requirements
The following packages are needed in order to run our code without error:
- python >= 3.8
- scipy>=1.4.1
- torch>=1.7.0
- torchvision>=0.8.1
- tqdm>=4.41.0
- numpy 
- matplotlib
- sklearn
- cuda 10.1

If you encounter installation issues, we recommend you to use our notebooks on kaggle or google colab.

## Download Optimal models

To reproduce our results you will need the optimal model checkpoint for every (optimizer,dataset) pair. The files are too large to fit on this repo, thus you can download them on this drive : https://drive.google.com/drive/u/2/folders/1CjqCi79LDUko2meM6F5VNB78pg1Cz1N0 using your epfl login. If you encounter download or access issues, please contact me at jeremy.baffou@epfl.ch.

## Reproducibility

The reproducibility is at the heart of all the scripts we produced. We followed gold standard guidelines of pytorch reproducibility : https://pytorch.org/docs/stable/notes/randomness.html. However we did not deactivate the randomness of several algorithms (such as CUDA). It may induce some slight variability in the results but not of big magnitude. The seeds can be changed for any script but by default it will be 2022, and you should use this one to reproduce our results (note that as it is set by default you do not need to specify it in our scripts).

## Notebook vs Scripts

We provide both notebooks and scripts for every tasks, except for the train vs validation plot where only the script exists. The figures and tests results have been generated using notebooks for display convenience. If you choose the notebook option, we advise you to run them on kaggle or google colab.  

Each script comes with its helper section that you can disply using: python "script_name.py" -h. We give here a brief description of each script:

- InitModelsPipeline.py : Create the initial models for each datasets based on several seeds.
- TrainingPipeline.py : Pipeline for model training given a dataset, an optimizer and its parameters.
- TestingPipeline.py : Pipeline to test a model against a given dataset which will be augmented using the paper's procedure.
- WeightsReport.py : Create the weights report for trajectories and frequency content of both the first CNN layer and linear layer.
- ValidationVSTraining.py : Create a plot of the training and validation accuracy on a given dataset for the four optimizer described in the paper.

The .py files model.py and nb_optimizers.py are helpers module used extensively accros our different pipelines.

## Figures

In the figures folder you will find all the (optimizer,dataset) pairs with their corresponding weights reports (both frequency content and trajectories). They have been produced using the notebook "Weights Visualization.ipynb". There is an additionnal folder named "validation_vs_train" which contains the plot and data for the training versus validation accuracy on CIFAR10 of the different optimizers.

