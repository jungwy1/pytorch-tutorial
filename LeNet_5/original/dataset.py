import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader


# Same preprocessing as the paper
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    # 2 pixel zero padding is used in the paper. (28x28 -> 32x32)
    v2.Pad(2),
    # normalization with range [-0.1, 1.175] is used in the paper.
    v2.Lambda(lambda x: x * 1.275 - 0.1),
])


def get_dataloaders(batch_size=64, path2data="../data"):
    training_data = torchvision.datasets.MNIST(
        root=path2data, train=True, download=True, transform=transform
    )
    test_data = torchvision.datasets.MNIST(
        root=path2data, train=False, download=True, transform=transform
    )
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader
