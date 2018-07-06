from typing import List, Tuple, Dict

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

####
# class used for cifar mnist coco data loading
# mainly deals with situation where in mainland china torchvision api is too slow
# with this api any user in mainland china can download the data from the web browser
# and use the pytorch api
####
class I_tv_loader:
    def __init__(self, root_dir: str, batch_size: int, n_workers: int):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.n_workers = n_workers

    @property
    def labels(self):
        raise NotImplementedError
    
    @property
    def train_loader(self):
        raise NotImplementedError

    @property
    def validation_loader(self):
        raise NotImplementedError

    @property
    def test_loader(self):
        raise NotImplementedError

class cifar10_loader(I_tv_loader):
    def __init__(self, root_dir: str, batch_size: int, n_workers: int):
        super(cifar10_loader, self).__init__(
                root_dir, batch_size, n_workers)
        transform_ = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.trainset = torchvision.datasets.CIFAR10(
            root=root_dir, train=True, 
            download=False, transform=transform_)
        self.testset = torchvision.datasets.CIFAR10(
            root=root_dir, train=False,
            download=False, transform=transform_)
        self.batch_size = batch_size

    @property
    def labels(self) -> List[str]:
        classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return classes
    
    @property
    def train_loader(self):
        trainloader = torch.utils.data.DataLoader(
                self.trainset, batch_size=self.batch_size, 
                shuffle=True, num_workers=self.n_workers)
        return trainloader

    @property
    def validation_loader(self):
        raise NotImplementedError

    @property
    def test_loader(self):
        testloader = torch.utils.data.DataLoader(
                self.testset, batch_size=self.batch_size,
                shuffle=False, num_workers=self.n_workers)
        return testloader

    @property
    def n_train_samples(self) -> int:
        return len(self.trainset)

    @property
    def n_test_samples(self) -> int:
        return len(self.testset)

    @property
    def n_iter_per_epoch(self) -> int:
        return self.n_train_samples // self.batch_size   

class mnist_loader(I_tv_loader):
    def __init__(self, root_dir: str, batch_size: int, n_workers: int):
        super(mnist_loader, self).__init__(
                root_dir, batch_size, n_workers)
        transform_ = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.trainset = torchvision.datasets.MNIST(
            root=root_dir, train=True,
            download=False, transform=transform_)
        self.testset = torchvision.datasets.MNIST(
            root=root_dir, train=False,
            download=False, transform=transform_)
        self.batch_size = batch_size

    @property
    def labels(self) -> List[str]:
        classes = [i for i in range(9)]    
        return classes
    
    @property
    def train_loader(self):
        trainloader = torch.utils.data.DataLoader(
                self.trainset, batch_size=self.batch_size, 
                shuffle=True, num_workers=self.n_workers)
        return trainloader

    @property
    def validation_loader(self):
        raise NotImplementedError

    @property
    def test_loader(self):
        testloader = torch.utils.data.DataLoader(
                self.testset, batch_size=self.batch_size,
                shuffle=False, num_workers=self.n_workers)
        return testloader

    @property
    def n_train_samples(self) -> int:
        return len(self.trainset)

    @property
    def n_test_samples(self) -> int:
        return len(self.testset)

    @property
    def n_iter_per_epoch(self) -> int:
        print("n_train {} batch_size {}".format(
            self.n_train_samples, self.batch_size))
        return self.n_train_samples//self.batch_size   

if __name__ == "__main___":
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.CIFAR10(root='/home/bigtree/PycharmProjects/torchloop/data/', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='/home/bigtree/PycharmProjects/torchloop/data/', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
