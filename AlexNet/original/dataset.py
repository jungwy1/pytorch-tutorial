import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader

# ============================================================================
# Preprocessing / augmentation vs. the AlexNet paper
# ----------------------------------------------------------------------------
# SAME as paper:
#   - Downsample to 256x256: resize shorter side to 256, then center-crop 256
#     (Section 2).
#   - Train augmentation form 1: random square crop + horizontal flip, taken
#     from the 256x256 image (Section 4.1, "translations and reflections").
#     Uses a fixed-size RandomCrop (position only), NOT RandomResizedCrop.
#
# DIFFERENT from paper:
#   - Input size 227, not 224. The paper text says 224, but 11x11/stride-4
#     conv1 only yields a clean 55x55 map from 227. (Well-known 224-vs-227 issue.)
#   - Normalize uses (x - mean) / std with ImageNet stats. The paper subtracts
#     only the per-pixel mean (no std division). mean/std is the modern default.
#   - Augmentation form 2 (PCA / "fancy PCA" color jitter, Section 4.1) is OMITTED.
#     It gives ~1% top-1 gain but is not built into torchvision. May add later.
#   - Evaluation uses a single CenterCrop(227). The paper averages 10 crops
#     (4 corners + center, each flipped) at test time. Simplified here.
#   - Dataset is Imagenette (10 classes, subset), not full ImageNet (1000).
# ============================================================================

# Common preprocessing: turn any image into a fixed 256x256 (paper Section 2)
preprocess = [
    v2.ToImage(),                       # PIL -> tensor
    v2.Resize(256, antialias=True),     # shorter side -> 256 (keep aspect ratio)
    v2.CenterCrop(256),                 # take center 256x256
]

# Train: random augmentation (paper Section 4.1, form 1)
train_transform = v2.Compose(preprocess + [
    v2.RandomCrop(227),                 # random 227 patch = translation aug
    v2.RandomHorizontalFlip(),          # 50% left-right mirror = reflection aug
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
    # NOTE: paper also applies PCA color augmentation here (omitted for now).
])

# Val: deterministic (single center crop; <-> paper uses 10-crop averaging)
val_transform = v2.Compose(preprocess + [
    v2.CenterCrop(227),                 # always center 227 (no randomness)
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

def get_dataloaders(batch_size = 64, path2data = "../data"):
    training_data = torchvision.datasets.Imagenette(
        root=path2data, 
        split="train", 
        size="320px", 
        download=True, 
        transform=train_transform
    )
    test_data = torchvision.datasets.Imagenette(
        root=path2data, 
        split="val", 
        size = "320px", 
        download=True, 
        transform=val_transform
    )
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader

# validation code for the above code
if __name__ == "__main__":
    train_ds = torchvision.datasets.Imagenette(root="../data", split="train",
                                            size="320px", download=True,
                                            transform=train_transform)
    img, label = train_ds[0]
    print(img.shape)                # expected size: torch.Size([3, 227, 227])
    mean, std = torch.zeros((1,)), torch.zeros((1,))
    for i in range(10):
        img, label = train_ds[i]
        mean += img.mean()
        std += img.std()
    mean /= 10
    std /= 10
    print(img.mean(), img.std())   # expected output: 0, 1

    train_dl, test_dl = get_dataloaders()
    X, y = next(iter(train_dl))
    print(X.shape, y.shape)   # expected: torch.Size([64, 3, 227, 227]) torch.Size([64])
