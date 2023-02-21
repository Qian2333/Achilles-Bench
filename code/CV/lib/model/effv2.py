import torch
import torchvision.models as models
import torch.nn as nn


class Effnet(torch.nn.Module):
    def __init__(self, num_cls=10):
        super(Effnet, self).__init__()
        self.model = models.efficientnet_v2_s(models.efficientnet.EfficientNet_V2_S_Weights)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_cls),
        )
        # self.fc = torch.nn.Linear(1000, num_cls)

    def forward(self, x):
        x = self.model(x)
        # x = self.fc(x)
        return x
