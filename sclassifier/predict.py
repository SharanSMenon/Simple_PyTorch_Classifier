import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image
from io import BytesIO
import requests


def apply_test_transforms(inp):
    trans = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return trans(inp)


def load_model_from_weights(weights_dir, num_classes):
    """Loads model from weight dir

    Args:
        weights_dir (str]): path to weights file
    """

    model = torchvision.models.densenet121(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False
    n_inputs = model.classifier.in_features
    last_layer = torch.nn.Linear(n_inputs, num_classes)
    model.classifier = last_layer
    model.load_state_dict(torch.load(weights_dir))
    return model


def load_image(image_dir, url=False):
    if not url:
        img = Image.open(image_dir)
    else:
        response = requests.get(image_dir)
        img = Image.open(BytesIO(response.content))
    return img


def predict(model, filepath, classes, url=False, device="cpu"):
    im = load_image(filepath, url=url)
    im_as_tensor = apply_test_transforms(im)
    minibatch = torch.stack([im_as_tensor])
    minibatch = minibatch.to(device)
    pred = model(minibatch)
    _, classnum = torch.max(pred, 1)
    return classes[classnum]
