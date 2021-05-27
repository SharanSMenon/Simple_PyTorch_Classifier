import torch
from torchvision import models


def create_model(num_classes, pretrained=True):
    model = models.densenet121(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = False
    n_inputs = model.classifier.in_features
    last_layer = torch.nn.Linear(n_inputs, num_classes)
    model.classifier = last_layer
    return model
