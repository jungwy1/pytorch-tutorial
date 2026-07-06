import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

from model import AlexNet
from dataset import get_dataloaders
from engine import test_loop

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
    save_path = "pruning_curve_no_finetune.png"
    if IN_COLAB:
        print("start pruning in COLAB")
        weight_path = "/content/drive/MyDrive/alexnet_weights.pth"
    else:
        print("start pruning in LOCAL")
        weight_path = "../original/alexnet_weights.pth"

    # ---- Reproducibility (same init every run / same init on Colab) ----
    torch.manual_seed(42)

    # ---- Data ----
    train_dataloader, test_dataloader = get_dataloaders(batch_size=128, path2data=path2data)

    # ----- Baseline load & val -----
    model = AlexNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(weight_path, map_location=device))
    base_acc = test_loop(test_dataloader, model, loss_fn, device)

    # ----- Pruning -----
    sparsities, accuracies = [0], [base_acc]
    for amount in [0.5, 0.7, 0.9, 0.95]:
        model = AlexNet().to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        params = [(m, "weight") for m in model.model
                if isinstance(m, (nn.Conv2d, nn.Linear))]
        prune.global_unstructured(
            params, pruning_method=prune.L1Unstructured, amount=amount
        )
        zeros = sum((m.weight == 0).sum().item() for m, _ in params)
        total = sum(m.weight.numel() for m, _ in params)
        print(f"sparsity: {100*zeros/total:.1f}%")
        acc = test_loop(test_dataloader, model, loss_fn, device)
        sparsities.append(amount * 100)
        accuracies.append(acc)
    
    # ----- Plot the results -----
    plt.figure(figsize=(7, 5))
    plt.plot(sparsities, accuracies, marker='o', linewidth=2,
             color="#2563eb", label="no fine-tune")
    plt.axhline(10, color="gray", linestyle="--", linewidth=1,
                label="random (10%)")
    for x, y in zip(sparsities, accuracies):
        plt.annotate(f"{y:.1f}%", (x,y), textcoords="offset points",
                     xytext=(0,8), ha="center", fontsize=9)
    plt.xlabel("Sparsity (%)")
    plt.ylabel("Accuracy (%)")
    plt.title("AlexNet magnitude pruning (no fine-tune)")
    plt.ylim(0, 85)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)   # ← PNG로 저장
    plt.show()
if __name__ == "__main__":
    main()