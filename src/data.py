"""Fashion-MNIST data loading via torchvision."""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

MEAN, STD = (0.2860,), (0.3530,)


def get_transforms(train: bool):
    t = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if train:
        t.insert(0, transforms.RandomHorizontalFlip())
    return transforms.Compose(t)


def get_loaders(data_dir="data", batch_size=128, val_size=5000, seed=42, num_workers=2):
    """Return train, validation and test DataLoaders."""
    full_train = datasets.FashionMNIST(data_dir, train=True, download=True, transform=get_transforms(True))
    test = datasets.FashionMNIST(data_dir, train=False, download=True, transform=get_transforms(False))
    g = torch.Generator().manual_seed(seed)
    train, val = random_split(full_train, [len(full_train) - val_size, val_size], generator=g)
    kw = dict(batch_size=batch_size, num_workers=num_workers)
    return (DataLoader(train, shuffle=True, **kw),
            DataLoader(val, shuffle=False, **kw),
            DataLoader(test, shuffle=False, **kw))
