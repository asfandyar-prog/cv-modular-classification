from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def make_dataloaders(
    name,
    batch_size=128,
    num_workers=2,
    data_dir="./data"
):
    name = name.lower()

    if name == "cifar10":
        num_classes = 10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)

        train_tfms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        val_tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_ds = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=train_tfms
        )

        val_ds = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=val_tfms
        )

    elif name == "cifar100":
        num_classes = 100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

        train_tfms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        val_tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_ds = datasets.CIFAR100(
            root=data_dir,
            train=True,
            download=True,
            transform=train_tfms
        )

        val_ds = datasets.CIFAR100(
            root=data_dir,
            train=False,
            download=True,
            transform=val_tfms
        )

    elif name == "stl10":
        num_classes = 10
        mean = (0.4467, 0.4398, 0.4066)
        std = (0.2241, 0.2215, 0.2239)

        train_tfms = transforms.Compose([
            transforms.Resize(96),
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        val_tfms = transforms.Compose([
            transforms.Resize(96),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_ds = datasets.STL10(
            root=data_dir,
            split="train",
            download=True,
            transform=train_tfms
        )

        val_ds = datasets.STL10(
            root=data_dir,
            split="test",
            download=True,
            transform=val_tfms
        )

    else:
        raise ValueError(f"Unknown dataset: {name}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, num_classes

