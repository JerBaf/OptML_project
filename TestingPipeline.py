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
        opts, args = getopt.getopt(argv,"h",["dataset=","batch_size","model=","seed="])
    except getopt.GetoptError:
        print(argv)
        print("Error in inputs, please be sure to provide at least the path to the control file.")
        sys.exit(2)
    dataset = "CIFAR10"
    batch_size = 1024
    seed = 2022
    model = ""
    for opt, arg in opts:
        if opt == "-h":
            print("\n")
            print("Help section:\n")
            print("This script is a pipeline to test a model with standard VGG architecture.")
            print("on the given Dataset with a few augmentations. It will be first tested on the basic test set.")
            print("Then it will be tested on three other datasets:")
            print("\t-The test set with gaussian noise (kernel_size=3,sigma=0.2)")
            print("\t-The test set with random affine transform (degree +- 20, translation +- 20%, scale +-15%")
            print("\t-The test set with random horizontal and vertical flip (both with probability 0.5).\n")
            print("Parameter:\n")
            print("-model: (String) Path to the model weights.\n")
            print("Keyword Arguments:\n")
            print("-dataset: (String) Name of the dataset. Please choose among:")
            print("\tCIFAR10, CIFAR100, MNIST or FashionMNIST. (default=CIFAR10)")
            print("-seed: (int) Seed to use for reproducibility (default=2022)\n")
            sys.exit(0)
        elif opt == "--dataset":
            dataset = arg
        elif opt == "--seed":
            seed = int(arg)
        elif opt == "--model":
            model = arg
    # Inputs Assertion
    if batch_size < 0:
        print("The minimum batch size is 0.")
        sys.exit(1)
    if dataset not in ["CIFAR10","CIFAR100","MNIST","FashionMNIST"]:
        print("The given data set name is not supported. Please select among CIFAR10,CIFAR100,MNIST or FashionMNIST.")
        sys.exit(1)
    if not os.path.isfile(model):
        print("The input model file does not exist, please provide an existing file")
        sys.exit(1)
    ### Testing Pipeline
    testing_pipeline(dataset,model,seed=seed,batch_size=batch_size)
    print("\nTesting over.")

### Helpers

def seed_worker(worker_id):
    """ Function for dataloader and worker reproducibility. """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def test_step(model,test_dataloader,device,criterion):
    """ Single test step with accuarcy evaluation. """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(test_dataloader,leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    accuracy = 100.*correct/total
    return accuracy

def testing_pipeline(dataset,model_path,seed=2022,batch_size=1024):
    """ Test the model on the given dataset with several augmentations. """
    ### Setup
    ssl._create_default_https_context = ssl._create_unverified_context
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_per_channel, std_per_channel),
      ])
    noise_transform = transforms.Compose([
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
        transforms.Normalize(mean_per_channel, std_per_channel),
      ])
    affine_transform = transforms.Compose([
        transforms.RandomAffine(20,translate=(0.2,0.2),scale=(0.85,1.15)),
        transforms.ToTensor(),
        transforms.Normalize(mean_per_channel, std_per_channel),
      ])
    flip_transform = transforms.Compose([
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean_per_channel, std_per_channel),
      ])
  ### Dataset Creation
    if dataset == "CIFAR10":
        dataset_test = torchvision.datasets.CIFAR10("data/",transform=base_transform,train=False)
        dataset_noise = torchvision.datasets.CIFAR10("data/",transform=noise_transform,train=False)
        dataset_affine = torchvision.datasets.CIFAR10("data/",transform=affine_transform,train=False)
        dataset_flip = torchvision.datasets.CIFAR10("data/",transform=flip_transform,train=False)
    elif dataset == "CIFAR100":
        dataset_test = torchvision.datasets.CIFAR100("data/",transform=base_transform,train=False)
        dataset_noise = torchvision.datasets.CIFAR100("data/",transform=noise_transform,train=False)
        dataset_affine = torchvision.datasets.CIFAR100("data/",transform=affine_transform,train=False)
        dataset_flip = torchvision.datasets.CIFAR100("data/",transform=flip_transform,train=False)
    elif dataset == "MNIST":
        dataset_test = torchvision.datasets.MNIST("data/",transform=base_transform,train=False)
        dataset_noise = torchvision.datasets.MNIST("data/",transform=noise_transform,train=False)
        dataset_affine = torchvision.datasets.MNIST("data/",transform=affine_transform,train=False)
        dataset_flip = torchvision.datasets.MNIST("data/",transform=flip_transform,train=False)
    elif dataset == "FashionMNIST":
        dataset_test = torchvision.datasets.FashionMNIST("data/",transform=base_transform,train=False)
        dataset_noise = torchvision.datasets.FashionMNIST("data/",transform=noise_transform,train=False)
        dataset_affine = torchvision.datasets.FashionMNIST("data/",transform=affine_transform,train=False)
        dataset_flip = torchvision.datasets.FashionMNIST("data/",transform=flip_transform,train=False)
    ### Dataloaders creation
    dataloader_test = DataLoader(dataset_test,batch_size=batch_size,pin_memory=True,
                                  worker_init_fn=seed_worker, generator=g, shuffle=True)
    dataloader_noise = DataLoader(dataset_noise,batch_size=batch_size,pin_memory=True,
                                  worker_init_fn=seed_worker, generator=g, shuffle=True)
    dataloader_affine = DataLoader(dataset_affine,batch_size=batch_size,pin_memory=True,
                                  worker_init_fn=seed_worker, generator=g, shuffle=True)
    dataloader_flip = DataLoader(dataset_flip,batch_size=batch_size,pin_memory=True,
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
    init_checkpoint = torch.load(model_path)
    state_dict = init_checkpoint["model"]
    formated_state_dict = dict()
    for key, value in state_dict.items():
        formated_state_dict[key.split("module.")[-1]] = value
    model.load_state_dict(formated_state_dict)
    model.to(device)
    if device.type == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    ### Old Validation Accuracy
    validation_accuracy = init_checkpoint["accuracy"]
    print(f"\nThe model has a validation accuracy of {validation_accuracy:.5f}%")
    ### Criterion Definition
    criterion = nn.CrossEntropyLoss()
    ### Testing steps
    base_acc = test_step(model,dataloader_test,device,criterion)
    noise_acc = test_step(model,dataloader_noise,device,criterion)
    affine_acc = test_step(model,dataloader_affine,device,criterion)
    flip_acc = test_step(model,dataloader_flip,device,criterion)
    print("The model has the following perfomances:")
    print(f"-\tAccuracy on the standard test set: {base_acc:.5f}%")
    print(f"-\tAccuracy on the noisy test set: {noise_acc:.5f}%")
    print(f"-\tAccuracy on the affine transformed test set: {affine_acc:.5f}%")
    print(f"-\tAccuracy on the flipped test set: {flip_acc:.5f}%")
    return base_acc, noise_acc, affine_acc, flip_acc

if __name__ == "__main__":
    main(sys.argv[1:])