import torch
import torch.nn as nn

from model import LeNet5
from dataset import get_dataloaders
from engine import train_loop, test_loop


# ---- Hyperparameters ----
batch_size = 64
learning_rate = 1e-3
epochs = 5


def main():
    # ---- Device ----
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")

    # ---- Reproducibility (same init every run / same init on Colab) ----
    torch.manual_seed(42)

    # ---- Data ----
    train_dataloader, test_dataloader = get_dataloaders(batch_size=batch_size)

    # ---- Model / loss / optimizer ----
    model = LeNet5().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ---- Train ----
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn, device)
    print("Done!")

    # ---- Save weights ----
    torch.save(model.state_dict(), "lenet5_weights.pth")
    print("Saved to lenet5_weights.pth")


if __name__ == "__main__":
    main()
