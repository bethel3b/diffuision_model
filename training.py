from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
from torch.utils.data import DataLoader
from unet_model import UNet
from torch.nn import MSELoss
from torch.optim import Adam
from trainer import Trainer



class Training:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.load_hyperparameters()
        

    def load_hyperparameters(self):
        self.batch_size = 100
        self.timestep = 1000
        self.epoch = 5
        self.lr_rate = 0.0001

    def get_cifar10(self):
        self.train_data = datasets.CIFAR10(
            root = "data",
            download=True,
            train=True, 
            transform=ToTensor()
        )
        self.test_data = datasets.CIFAR10(
            root = "data",
            download=True,
            train=False, 
            transform=ToTensor()
        )
    def dataloader(self):
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False)
        return

    def initialize_model(self):
        self.model = UNet()
        self.loss = MSELoss()
        self.optimizer = Adam(params=self.model, lr=self.lr_rate)
        return

    
    
    def run(self):
        self.get_cifar10()
        self.dataloader()
        self.initialize_model()

        training_args = {"model":self.model,
                         "train_loader": self.train_loader,
                         "loss": self.loss,
                         "optimizer": self.optimizer,
                         "device": self.device}
        Trainer(args=training_args)
        return 