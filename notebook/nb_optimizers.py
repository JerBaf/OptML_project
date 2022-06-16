import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod

class optimizer:
    def __init__(self, parameters, device):
        self.parameters = list(parameters)
        self.device = device

    @abstractmethod
    def step(self):
        pass

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None


class SGDOptimizer(optimizer):
    def __init__(self, parameters, device, args):
        super().__init__(parameters, device)
        self.learning_rate = args["learning_rate"]

    def step(self):
        for p in self.parameters:
            p.data -= p.grad * self.learning_rate


class MomentumSGDOptimizer(optimizer):
    def __init__(self, parameters, device, args):
        super().__init__(parameters, device)
        self.learning_rate = args["learning_rate"]
        self.rho = args["rho"]
        self.m = None

    def step(self):
        if self.m is None:
            self.m = [torch.zeros(p.size()).to(self.device) for p in self.parameters]

        for i, p in enumerate(self.parameters):
            self.m[i] = self.rho * self.m[i] + p.grad
            p.grad = self.learning_rate * self.m[i]
            p.data -= p.grad


class RMSPropOptimizer(optimizer):
    def __init__(self, parameters, device, args):
        super().__init__(parameters, device)
        self.tau = args["tau"]
        self.learning_rate = args["learning_rate"]
        self.r = None
        self.delta = args["delta"]

    def step(self):
        if self.r is None:
            self.r = [torch.zeros(p.size()).to(self.device) for p in self.parameters]

        for i, p in enumerate(self.parameters):
            grad = p.grad
            self.r[i] = self.r[i] * self.tau + (1 - self.tau) * grad * grad
            p.data -= self.learning_rate / (self.delta + torch.sqrt(self.r[i])) * grad


class AMSgradOptimizer(optimizer):
    def __init__(self, parameters, device, args):
        super().__init__(parameters, device)
        self.beta1 = args["beta1"]
        self.beta2 = args["beta2"]
        self.learning_rate = args["learning_rate"]
        self.delta = args["delta"]
        self.iteration = None
        self.m1 = None
        self.m2 = None
        self.m2_max = None 

    def step(self):

        if self.m1 is None:
            self.m1 = [torch.zeros(p.grad.size()).to(self.device) for p in self.parameters]
        if self.m2 is None:
            self.m2 = [torch.zeros(p.grad.size()).to(self.device) for p in self.parameters]
        if self.m2_max is None:
            self.m2_max = [torch.zeros(p.grad.size()).to(self.device) for p in self.parameters]
        if self.iteration is None:
            self.iteration = 1

        for i, p in enumerate(self.parameters):
            grad = p.grad
            self.m1[i] = self.m1[i] * self.beta1 + (1 - self.beta1) * grad
            self.m2[i] = self.m2[i] * self.beta2 + (1 - self.beta2) * grad * grad
            m1_hat = self.m1[i] / (1 - self.beta1 ** self.iteration)
            m2_hat = self.m2[i] / (1 - self.beta2 ** self.iteration)
            self.m2_max[i] = torch.maximum(m2_hat, self.m2_max[i])
            p.data -= self.learning_rate * m1_hat / (self.delta + torch.sqrt(self.m2_max[i]))

        self.iteration = self.iteration + 1


class AdagradOptimizer(optimizer):
    def __init__(self, parameters, device, args):
        super().__init__(parameters, device)
        self.learning_rate = args["learning_rate"]
        self.delta = args["delta"]
        self.r = None

    def step(self):
        if self.r is None:
            self.r = [torch.zeros(p.size()).to(self.device) for p in self.parameters]

        for i, p in enumerate(self.parameters):
            grad = p.grad
            self.r[i] = self.r[i] + grad * grad
            p.data -= self.learning_rate / (self.delta + torch.sqrt(self.r[i])) * grad


class ADAMOptimizer(optimizer):
    def __init__(self, parameters, device, args):
        super().__init__(parameters, device)
        self.beta1 = args["beta1"]
        self.beta2 = args["beta2"]
        self.learning_rate = args["learning_rate"]
        self.delta = args["delta"]
        self.iteration = None
        self.m = None
        self.v = None

    def step(self):
        if self.m is None:
            self.m = [torch.zeros(p.grad.size()).to(self.device) for p in self.parameters]
        if self.v is None:
            self.v = [torch.zeros(p.grad.size()).to(self.device) for p in self.parameters]
        if self.iteration is None:
            self.iteration = 1

        for i, p in enumerate(self.parameters):
            grad = p.grad
            self.m[i] = self.m[i] * self.beta1 + (1 - self.beta1) * grad
            self.v[i] = self.v[i] * self.beta2 + (1 - self.beta2) * grad * grad
            m_hat = self.m[i] / (1 - self.beta1 ** self.iteration)
            v_hat = self.v[i] / (1 - self.beta2 ** self.iteration)
            p.data -= self.learning_rate * m_hat / (self.delta + torch.sqrt(v_hat))

        self.iteration = self.iteration + 1


def createOptimizer(device, args, model):
    p = model.parameters()
    if args["optimizer"] == "sgd":
        return SGDOptimizer(p, device, args)
    elif args["optimizer"] == "momentumsgd":
        return MomentumSGDOptimizer(p, device, args)
    elif args["optimizer"] == "adagrad":
        return AdagradOptimizer(p, device, args)
    elif args["optimizer"] == "adam":
        return ADAMOptimizer(p, device, args)
    elif args["optimizer"] == "rmsprop":
        return RMSPropOptimizer(p, device, args)
    elif args["optimizer"] == "amsgrad":
        return AMSgradOptimizer(p, device, args)
    else:
        raise NotImplementedError(f"Unknown optimizer {args['optimizer']}")