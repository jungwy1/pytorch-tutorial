import torch
import torch.nn as nn

from model import AlexNet
from dataset import get_dataloaders
from engine import train_loop, test_loop

# ----- Hyperparameters -----
weight_decay = 5e-4
batch_size = 128
learning_rate = 1e-4
epochs = 20
step_size = 10
gamma = 0.1

def main():
    # ---- Path ----
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    
    if IN_COLAB:
        print("start training in COLAB")
        path2data = "data"
        save_path = "/content/drive/MyDrive/alexnet_weights.pth"
    else:
        print("start training in LOCAL")
        path2data = "../data"
        save_path = "alexnet_weights.pth"
    # ---- Device ----
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")

    # ---- Reproducibility (same init every run / same init on Colab) ----
    torch.manual_seed(42)

    # ---- Data ----
    train_dataloader, test_dataloader = get_dataloaders(batch_size=batch_size, path2data=path2data)

    # ---- Model / loss / optimizer ----
    model = AlexNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # ---- Train ----
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        print(f"lr: {scheduler.get_last_lr()[0]}")
        test_loop(test_dataloader, model, loss_fn, device)
        scheduler.step()
        # ---- Save weights ----
        torch.save(model.state_dict(), save_path)
    print("Done!")
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()