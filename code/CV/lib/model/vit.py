import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ViTFeatureExtractor, ViTModel, ViTForImageClassification


class Vit(nn.Module):
    def __init__(self, num_cls=10, model='google/vit-base-patch16-224'):
        super(Vit, self).__init__()

        # self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTModel.from_pretrained(model)

        self.fc = nn.Linear(768, num_cls)

    def forward(self, x):
        # print(x.shape)
        # exit(0)
        # x = self.feature_extractor(images=x.cpu(), return_tensors="pt").cuda()
        outputs = self.model(x)
        # print(outputs.last_hidden_state.shape)
        last_hidden_states = outputs.last_hidden_state[:, 0, :]
        # print(last_hidden_states.shape)
        # exit(0)
        return self.fc(last_hidden_states)
