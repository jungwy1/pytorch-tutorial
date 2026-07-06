import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
from adjustText import adjust_text

from model import AlexNet
from dataset import get_dataloaders
from engine import train_loop, test_loop

# ----- Hyperparameters -----
weight_decay = 5e-4
batch_size = 128
learning_rate = 4e-5
finetune_epochs = 8
gamma = 0.1

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
        weight_path = "/content/drive/MyDrive/alexnet_weights.pth"
        save_path = "/content/drive/MyDrive/pruning_curve_with_finetune.png"
    else:
        print("start pruning in LOCAL")
        weight_path = "../original/alexnet_weights.pth"
        save_path = "pruning_curve_with_finetune.png"

    # ---- Reproducibility (same init every run / same init on Colab) ----
    torch.manual_seed(42)

    # ---- Data ----
    train_dataloader, test_dataloader = get_dataloaders(batch_size=batch_size, path2data=path2data)

    # ----- Baseline load & val -----
    model = AlexNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(weight_path, map_location=device))
    base_acc = test_loop(test_dataloader, model, loss_fn, device)

    # ----- Pruning -----
    sparsities, accuracies = [0], [base_acc]
    for amount in [i / 100 for i in range(75, 100, 5)]:
        # weight load
        model = AlexNet().to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        # prune (masking)
        params = [(m, "weight") for m in model.model
                if isinstance(m, (nn.Conv2d, nn.Linear))]
        prune.global_unstructured(
            params, pruning_method=prune.L1Unstructured, amount=amount
        )
        # fine-tune
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for e in range(finetune_epochs):
            train_loop(train_dataloader, model, loss_fn, optimizer, device)
        # test
        zeros = sum((m.weight == 0).sum().item() for m, _ in params)
        total = sum(m.weight.numel() for m, _ in params)
        print(f"sparsity: {100*zeros/total:.1f}%")
        acc = test_loop(test_dataloader, model, loss_fn, device)
        sparsities.append(amount * 100)
        accuracies.append(acc)
    
    # ----- no fine-tune results -----
    nf_sparsities = [0, 75, 80, 85, 90, 95]
    nf_accuracies = [77.0, 69.9, 62.3, 45.3, 17.1, 12.4]


    # ----- Plot the results -----
    plt.figure(figsize=(11, 5))
    # no fine-tune
    plt.plot(nf_sparsities, nf_accuracies, marker='o', linewidth=2,
            color="gray", alpha=0.6, label="no fine-tune")
    # with fine-tune
    plt.plot(sparsities, accuracies, marker='o', linewidth=2,
            color="#2563eb", label="with fine-tune")
    plt.axhline(10, color="gray", linestyle="--", linewidth=1,
                label="random (10%)") 
    plt.xlabel("Sparsity (%)")
    plt.ylabel("Accuracy (%)")
    plt.title("AlexNet magnitude pruning: fine-tune vs no fine-tune")
    plt.ylim(0, 85)
    plt.grid(True, alpha=0.3)
    plt.legend()
    texts = [plt.text(x, y, f"{y:.1f}%", fontsize=8) for x, y in zip(sparsities, accuracies)]
    adjust_text(texts)   # 겹치는 라벨을 자동으로 밀어냄
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)   # ← PNG로 저장
    plt.show()
if __name__ == "__main__":
    main()