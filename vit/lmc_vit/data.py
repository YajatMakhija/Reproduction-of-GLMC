"""CIFAR-10 / CIFAR-100 data loaders (shared by all scripts)."""

import torch
import torchvision
import torchvision.transforms as T

STATS = {
    "CIFAR-10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "CIFAR-100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}
NUM_CLASSES = {"CIFAR-10": 10, "CIFAR-100": 100}
_DATASET = {"CIFAR-10": torchvision.datasets.CIFAR10, "CIFAR-100": torchvision.datasets.CIFAR100}


def get_loaders(dataset="CIFAR-10", batch_size=128, root="./data", num_workers=4,
                augment=True, download=True):
    dataset = dataset.upper()
    if dataset not in STATS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    mean, std = STATS[dataset]
    cls = _DATASET[dataset]

    train_tf = T.Compose(([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    ] if augment else []) + [T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    trainset = cls(root=root, train=True, download=download, transform=train_tf)
    testset = cls(root=root, train=False, download=download, transform=test_tf)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=True)
    return trainloader, testloader, NUM_CLASSES[dataset]
