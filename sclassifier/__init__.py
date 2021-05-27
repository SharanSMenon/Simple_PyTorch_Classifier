import torch
from .model import create_model
from .dataloaders import get_three_loaders, get_train_val_test_loaders
from .predict import load_model_from_weights
from .train import train


def train_classifier_on_directory(data_dir: str, val=True, test=True,
                                  data_in_train_dir=False, N_EPOCHS=5):
    """
    Trains a classifier on a directory
    Args:
        data_dir: Directory of your data

    Returns: a model
    """
    classes = []
    val_loader = None
    test_loader = None
    if not data_in_train_dir:
        (train_loader, val_loader, test_loader), classes = get_three_loaders(data_dir)
    else:
        if val and test:
            loaders, classes = get_train_val_test_loaders(data_dir)
            train_loader = loaders[0]
            val_loader = loaders[1]
            test_loader = loaders[2]
        elif val and not test:
            loaders, classes = get_train_val_test_loaders(data_dir)
            train_loader = loaders[0]
            val_loader = loaders[1]
        else:
            loaders, classes = get_train_val_test_loaders(data_dir)
            train_loader = loaders[0]
            test_loader = loaders[1]

    if len(classes) == 0:
        raise FileNotFoundError("There are no classes found in your directory, please give a valid directory")
    classifier = create_model(len(classes), pretrained=True)
    train(N_EPOCHS, classifier, train_loader, val_loader=val_loader)
    return classifier
