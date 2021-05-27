from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import os


def get_three_loaders(data_dir: str, batch_size=64):
    """This function creates a train, test, and val loaders out of a single data directory,
    and also returns the classes

    Args:
        data_dir ([string]): path to data
    """
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])
    all_data = datasets.ImageFolder(data_dir, transform=transform)
    train_data_len = int(len(all_data) * 0.75)
    valid_data_len = int((len(all_data) - train_data_len) / 2)
    test_data_len = int(len(all_data) - train_data_len - valid_data_len)
    train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return (train_loader, val_loader, test_loader), all_data.classes


def get_train_val_test_loaders(data_dir: str, batch_size=64, val=True, test=True):
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])
    all_classes = []
    train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    loaders = [train_loader]
    all_classes.extend(train_data.classes)
    if val:
        val_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        all_classes.extend(val_data.classes)
        loaders.append(val_loader)
    if test:
        test_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        all_classes.extend(test_data.classes)
        loaders.append(test_loader)

    return loaders, list(set(all_classes))
