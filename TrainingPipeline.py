import torch
import torchvision
import ssl
import random
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import tqdm as tqdm
from torch.utils.data.sampler import SubsetRandomSampler
import os
import sys
import getopt
# Custom helpers
from model import *
import nb_optimizers as opt


def main(argv):
    # Parse arguments
    try:
        opts, args = getopt.getopt(argv,"h",["dataset=","max_epoch=","batch_size=","optimizer=", "lr=","output_name=","init_model=","seed="])
    except getopt.GetoptError:
        print(argv)
        print("Error in inputs, please be sure to provide at least the path to the control file.")
        sys.exit(2)
    dataset = "CIFAR10"
    max_epoch = 100
    batch_size = 1024
    optimizer = "sgd"
    seed = 2022
    lr = 5e-1
    output_name = ""
    init_model = ""
    for opt, arg in opts:
        if opt == "-h":
            print("\n")
            print("Help section:\n")
            print("This script is a pipeline to train a model with standard VGG architecture.")
            print("on the given Dataset.\n")
            print("Parameter:\n")
            print("-init_model: (String) Path to the initial model weights. Please select a path")
            print("\t to a model generated by the script init_model.py provided to ensure reproducibility.")
            print("-output_name: (String) Basename given to the output folder. Please select a name which is")
            print("\t not already in the working directory.\n")
            print("Keyword Arguments:\n")
            print("-dataset: (String) Name of the dataset. Please choose among:")
            print("\tCIFAR10, CIFAR100, MNIST or FashionMNIST. (default=CIFAR10)")
            print("-max_epoch: (int) Maximum number of epoch. (default=100)")
            print("-batch_size: (int) Size of the dataloader batches. (default=1024)")
            print("-optimizer: (String) Name of the optimizer. Please choose among:")
            print("\tsgd, momentumsgd, adam or rmsprop. (default=sgd)")
            print("-lr: (int) Learining rate for optimization. (default=5e-1)")
            print("-seed: (int) Seed to use for reproducibility (default=2022)\n")
            print("Outputs:\n")
            print("The script will output 3 different files:")
            print("-cnn_weights.npy: (Numpy binary) Array containing the weights of the first cnn layer.")
            print("-linear_weights.npy: (Numpy binary) Array containing the weights of the linear layer.")
            print("-ckpt.pth: (Pytorch checkpoint) Pytorch model with the best validation accuracy.")
            sys.exit(0)
        elif opt == "--dataset":
            dataset = arg
        elif opt == "--max_epoch":
            max_epoch = int(arg)
        elif opt == "--batch_size":
            batch_size = int(arg)
        elif opt == "--optimizer":
            optimizer = arg
        elif opt == "--lr":
            lr = int(arg)
        elif opt == "--output_name":
            output_name = arg
        elif opt == "--seed":
            seed = int(arg)
        elif opt == "--init_model":
            init_model = arg
    # Inputs Assertion
    if batch_size < 0:
        print("The minimum batch size is 0.")
        sys.exit(1)
    if optimizer not in ["sgd","momentumsgd","adam","rmsprop"]:
        print(optimizer)
        print("The given optimizer name is not supported. Please select among sgd,momentumsgd,adam or rmsprop.")
        sys.exit(1)
    if dataset not in ["CIFAR10","CIFAR100","MNIST","FashionMNIST"]:
        print("The given data set name is not supported. Please select among CIFAR10,CIFAR100,MNIST or FashionMNIST.")
        sys.exit(1)
    if lr < 0:
        print("The minimum learning rate is 0.")
        sys.exit(1)
    if max_epoch < 1:
        print("The minimum number of epoch is 1.")
        sys.exit(1)
    if not os.path.isfile(init_model):
        print("The input model file does not exist, please provide an existing file")
        sys.exit(1)
    if os.path.isdir(output_name):
        print(f"The folder {output_name} already exists, please delete it or provide another name")
        sys.exit(1)
    try:
        os.mkdir(output_name)
    except:
        print(f"Impossible to create output folder {output_name}. Please be sure of your permissions.")
        sys.exit(1)
    ### Optimizer setup
    optimizer_parameters = {"optimizer": optimizer, "learning_rate": lr, "rho": 0.9, "tau": 0.99, "delta": 1e-8, "beta1": 0.9,
                            "beta2": 0.999}
    ### Training Pipeline
    training_pipeline(dataset,init_model,optimizer_parameters,output_name,seed=seed,batch_size=batch_size,max_epoch=max_epoch)
    print("Training over, please check outputs for more details.")

### Helpers

def training_pipeline(dataset,init_model_pth,optimizer_parameters,basename,seed=2022,batch_size=1024,max_epoch=75):
    """ Pipeline used to train the model on the given dataset. """
    ### Device setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ssl._create_default_https_context = ssl._create_unverified_context
  ### Reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
  ### Download Datasets
    if dataset == "CIFAR10":
        dataset_train = torchvision.datasets.CIFAR10("data/",download=True)
        dataset_test = torchvision.datasets.CIFAR10("data/",download=True,train=False)
    elif dataset == "CIFAR100":
        dataset_train = torchvision.datasets.CIFAR100("data/",download=True)
        dataset_test = torchvision.datasets.CIFAR100("data/",download=True,train=False)
    elif dataset == "MNIST":
        dataset_train = torchvision.datasets.MNIST("data/",download=True)
        dataset_test = torchvision.datasets.MNIST("data/",download=True,train=False)
    elif dataset == "FashionMNIST":
        dataset_train = torchvision.datasets.FashionMNIST("data/",download=True)
        dataset_test = torchvision.datasets.FashionMNIST("data/",download=True,train=False)
    else:
        raise Exception("Unavailable dataset, please select among CIFAR10, CIFAR100, MNIST, FashionMNIST.")
  ### Compute initial Transform
    if dataset in ["CIFAR10","CIFAR100"]:
        mean_per_channel = tuple((dataset_train.data/255).mean(axis=(0,1,2)))
        std_per_channel = tuple((dataset_train.data/255).std(axis=(0,1,2)))
    else:
        mean_per_channel = (dataset_train.data.numpy()/255).mean()
        std_per_channel = (dataset_train.data.numpy()/255).std()
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean_per_channel, std_per_channel),
    ])
  ### Dataset Creation
    if dataset == "CIFAR10":
        dataset_train = torchvision.datasets.CIFAR10("data/",transform=transform)
        dataset_test = torchvision.datasets.CIFAR10("data/",transform=transform,train=False)
    elif dataset == "CIFAR100":
        dataset_train = torchvision.datasets.CIFAR100("data/",transform=transform)
        dataset_test = torchvision.datasets.CIFAR100("data/",transform=transform,train=False)
    elif dataset == "MNIST":
        dataset_train = torchvision.datasets.MNIST("data/",transform=transform)
        dataset_test = torchvision.datasets.MNIST("data/",transform=transform,train=False)
    elif dataset == "FashionMNIST":
        dataset_train = torchvision.datasets.FashionMNIST("data/",transform=transform)
        dataset_test = torchvision.datasets.FashionMNIST("data/",transform=transform,train=False)
  ### Validation Split
    train_sampler, val_sampler = get_samplers(dataset_train,g)
  ### Dataloaders creation
    dataloader_train = DataLoader(dataset_train,batch_size=batch_size,pin_memory=True,
                                  worker_init_fn=seed_worker, generator=g, sampler=train_sampler,
                                      )
    dataloader_val = DataLoader(dataset_train,batch_size=batch_size,pin_memory=True,
                                  worker_init_fn=seed_worker, generator=g, sampler=val_sampler,
                                      )
    dataloader_test = DataLoader(dataset_test,batch_size=batch_size,pin_memory=True,
                                  worker_init_fn=seed_worker, generator=g, shuffle=True)
  ### Model Creation
    if dataset == "CIFAR10":
        in_c, out_c = (3,10)
    elif dataset == "CIFAR100":
        in_c, out_c = (3,100)
    elif dataset == "MNIST":
        in_c, out_c = (1,10)
    elif dataset == "FashionMNIST":
        in_c, out_c = (1,10)
    model = VGG(in_c,out_c)
    init_checkpoint = torch.load(init_model_pth)
    model.load_state_dict(init_checkpoint['model_state_dict'])
    model.to(device)
    if device.type == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
  ### Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.createOptimizer(device, optimizer_parameters, model)
    scheduler = None
  ### Weights Collection
    cnn_layer_weights = []
    linear_layer_weights = []
  ### Saving names
    ckpt_name = basename
    weight_folder_name = basename
  ### Training Loop
    training_loop(max_epoch,dataloader_train,device,optimizer,criterion,model,dataloader_val,
                  ckpt_name,scheduler,cnn_layer_weights,linear_layer_weights,in_c)
  ### Weights Saving
    save_weights_for_viz(cnn_layer_weights,linear_layer_weights,weight_folder_name+"/")

def seed_worker(worker_id):
    """ Function for reproducibility with the workers. """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_samplers(train_dataset,generator,shuffle=True,val_ratio=0.1):
    """ Give the train and validation samplers for the dataloaders. """
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_ratio * num_train))
    if shuffle:
        np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx,generator=generator)
    val_sampler = SubsetRandomSampler(val_idx,generator=generator)
    return train_sampler, val_sampler

def collect_weights(cnn_weights_list,linear_weights_list,model,channels_nb=3):
    """ Collect weights of the first cnn layer and the linear layer for further evaluation. """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if m.in_channels == channels_nb:
                cnn_weights_list.append(m.weight.ravel().detach().cpu().numpy())
        elif isinstance(m, nn.Linear):
            linear_weights_list.append(m.weight.ravel().detach().cpu().numpy())

def train_step(model,train_dataloader,device,optimizer,criterion,epoch,cnn_layer_weights,linear_layer_weights,channels):
    """ Standard single training step. """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    idx = 0
    for inputs, targets in tqdm.tqdm(train_dataloader,leave=False):
        ### Collect Weights
        if (idx%4) == 0:
            collect_weights(cnn_layer_weights,linear_layer_weights,model,channels)
        ### Perform training
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        ### Compute Accuracy
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        idx += 1
    print(f"At end of epoch {epoch} we have average loss {train_loss/total:.5f} and average accuracy {correct/total:.5f}%")  

def validation_step(model,val_dataloader,device,criterion,best_acc,epoch,checkpoint_name="checkpoint"):
    """ Standard validation step, with saving of the model if it has achieved the best accuracy so far."""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(val_dataloader,leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    # Save checkpoint.
    accuracy = 100.*correct/total
    if accuracy > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'accuracy': accuracy,
            'epoch': epoch,
        }
        if not os.path.isdir(checkpoint_name):
            os.mkdir(checkpoint_name)
        torch.save(state, checkpoint_name+"/ckpt.pth")
        print(f"New optimal model at epoch {epoch} saved with validation accuracy {correct/total:.5f}%")
    else:
        print(f"Validation accuracy {correct/total:.5f}%")
    return accuracy

def training_loop(max_epoch,dataloader_train,device,optimizer,criterion,model,dataloader_val,
                  ckpt_name,scheduler,cnn_layer_weights,linear_layer_weights,channels):
    """ Loop for training, alternating between training and validation step. """
    best_accuracy = -1
    for epoch in tqdm.tqdm(range(max_epoch)):
        train_step(model,dataloader_train,device,optimizer,criterion,epoch,cnn_layer_weights,linear_layer_weights,channels)
        epoch_accuracy = validation_step(model,dataloader_val,device,criterion,best_accuracy,epoch,checkpoint_name=ckpt_name)
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
        if scheduler != None:
            scheduler.step()

def save_weights_for_viz(cnn_weights,linear_weights,basename):
    """ Save weights for further investigation. """
    cnn_file = open(basename+"cnn_weights.npy","wb")
    linear_file = open(basename+"linear_weights.npy","wb")
    np.save(cnn_file,cnn_weights)
    np.save(linear_file,linear_weights)
    cnn_file.close()
    linear_file.close()


if __name__ == "__main__":
    main(sys.argv[1:])