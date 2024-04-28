from torchvision.models.resnet import resnet50
from torch import nn
import torch


class Model_SimCLR(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model_SimCLR, self).__init__()

        list_modules = []
        for name, module in resnet50().named_children():
            if not isinstance(module, nn.Linear):
                list_modules.append(module)
        # encoder
        self.encoder = nn.Sequential(*list_modules)

        # projection head
        self.projection = nn.Sequential(
                                nn.Linear(2048, 512, bias=False), 
                                nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True), 
                                nn.Linear(512, feature_dim, bias=True))
    
    def encode(self, x):
        features = self.encoder(x)
        return torch.flatten(features, start_dim=1)

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.projection(feature)
        return feature, out
