import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os

from model import AlexNet
from dataset import get_dataloaders
from engine import train_loop, test_loop

# ----- Hyperparameters -----
weight_decay = 5e-4     # weight decay
batch_size = 128        # dataloader batch size
learning_rate = 4e-5    # learning rate
finetune_epochs = 8     # fine-tune epochs
amount = 0.9            # pruning rate

def main():
    # ----- Device -----
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")

    # ---- Path ----
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    
    path2data = "../data"
    if IN_COLAB:
        print("start pruning in COLAB")
        weight_dir = "/content/drive/MyDrive"
        save_dir = "/content/drive/MyDrive"
    else:
        print("start pruning in LOCAL")
        weight_dir = "../original"
        save_dir = "."
    weight_path = os.path.join(weight_dir, "alexnet_weights.pth")
    save_path = os.path.join(save_dir, "alexnet_pruned90.pth")

    # ---- Reproducibility (same init every run / same init on Colab) ----
    torch.manual_seed(42)

    # ---- Data ----
    train_dataloader, test_dataloader = get_dataloaders(batch_size=batch_size, path2data=path2data)

    # ----- Baseline load & val -----
    model = AlexNet().to(device)
    
    model.load_state_dict(torch.load(weight_path, map_location=device))

    # ----- Pruning -----
    loss_fn = nn.CrossEntropyLoss()
    # prune (masking)
    params = [(m, "weight") for m in model.model
            if isinstance(m, (nn.Conv2d, nn.Linear))]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    # fine-tune
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for e in range(finetune_epochs):
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
    # make pruning permanent
    for m, name in params:
        prune.remove(m, name)
    # test
    zeros = sum((m.weight == 0).sum().item() for m, _ in params)
    total = sum(m.weight.numel() for m, _ in params)
    print(f"sparsity: {100*zeros/total:.1f}%")
    acc = test_loop(test_dataloader, model, loss_fn, device)

    # ----- Save weights -----
    torch.save(model.state_dict(), save_path)
    
    # ----- Validation ------
    model2 = AlexNet().to(device)
    model2.load_state_dict(torch.load(save_path, map_location=device))
    print("reload check:")
    params2 = [(m, "weight") for m in model2.model
            if isinstance(m, (nn.Conv2d, nn.Linear))]
    zeros = sum((m.weight == 0).sum().item() for m, _ in params2)
    total = sum(m.weight.numel() for m, _ in params2)
    print(f"sparsity: {100*zeros/total:.1f}%")
    test_loop(test_dataloader, model2, loss_fn, device)

if __name__ == "__main__":
    main()