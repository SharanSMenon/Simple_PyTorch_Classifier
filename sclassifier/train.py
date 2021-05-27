import torch
from tqdm import tqdm
from torch import nn, optim
from .model import create_model
from .dataloaders import get_three_loaders
from .utils import get_device


def validate(classifier, criterion, valid_loader, device="cpu"):
    eval_acc = 0.0
    eval_loss = 0.0
    for images, labels in tqdm(valid_loader):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            output = classifier(images)
            loss = criterion(output, labels)

        acc = ((output.argmax(dim=1) == labels).float().mean())
        eval_acc += acc
        eval_loss += loss
    return eval_acc / len(valid_loader), eval_loss / len(valid_loader)


def train_one_epoch(classifier, optimizer, criterion, trainloader, device="cpu"):
    train_acc = 0.0
    train_loss = 0.0
    for images, labels in tqdm(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = classifier(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        acc = ((output.argmax(dim=1) == labels).float().mean())
        train_acc += acc
        train_loss += loss
    return train_acc / len(trainloader), train_loss / len(trainloader)


def train(N_EPOCHS, model, train_loader, val_loader=None, save=False, checkpoint=False):
    device = get_device()
    model = model.to(device)

    results = {
        "train_acc": [],
        "train_loss": [],
        "eval_acc": [],
        "eval_loss": []
    }

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters())

    for epoch in range(N_EPOCHS):
        train_acc, train_loss = train_one_epoch(model, optimizer, criterion,
                                                train_loader, device=device)
        print("")
        print(f"Epoch {epoch + 1} | Train Acc: {train_acc * 100} | Train Loss: {train_loss}")

        if val_loader:
            eval_acc, eval_loss = validate(model, criterion, val_loader, device=device)
            print("")
            print(f"\t Val Acc: {eval_acc * 100} | Val Loss: {eval_loss}")
        print("====" * 8)


if __name__ == "__main__":
    (train_loader, val_loader, test_loader), classes = get_three_loaders(
        "sample_data_dir")
    model = create_model(len(classes), pretrained=True)
