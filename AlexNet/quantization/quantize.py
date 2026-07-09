"""
OFFLINE compile step: calibrate + prepare AlexNet's classifier into integer artifacts,
then SAVE them. The saved file is the deployable model -- int8 weights + scales, with NO
float FC weights -- and is loaded by infer.py for integer-only inference.

Run this once. Pipeline:
    load float model -> calibrate on real batches (freeze scale/zp)
                     -> QLinear.prepare (quantize weights, fold constants)
                     -> torch.save the QLinear artifacts + input calibration + float conv
"""
import os
import torch

from model import AlexNet
from dataset import get_dataloaders
from quant_basics import compute_qparams
from quant_linear_layer import QLinear


class RangeTracker:
    """Accumulate global min/max of a tensor across calibration batches."""
    def __init__(self):
        self.min, self.max = float("inf"), float("-inf")

    def update(self, t):
        self.min = min(self.min, t.min().item())
        self.max = max(self.max, t.max().item())

    def qparams(self, n_bits=8):
        return compute_qparams(self.min, self.max, n_bits)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"calibration: features on {device}")

    # ---- paths: local vs Colab ----
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
        
    if IN_COLAB:
        weight_path = "/content/drive/MyDrive/alexnet_pruned90.pth"
        save_path = "/content/drive/MyDrive/alexnet_int8_classifier.pth"
        path2data = "/content/data"
    else:
        weight_path = "../pruning/alexnet_pruned90.pth"
        save_path = "alexnet_int8_classifier.pth"
        path2data = "../data"

    # ---- load the trained float model ----
    model = AlexNet().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()                                     # dropout OFF

    train_loader, _ = get_dataloaders(batch_size=64, path2data=path2data)

    features = model.model[:14]                      # conv/relu/pool ... Flatten
    fc1, fc2, fc3 = model.model[14], model.model[17], model.model[20]

    # ================================ CALIBRATION ================================
    # run a few real batches; collect the range at each FC boundary
    n_calib_batches = 10
    tr_x, tr_h1, tr_h2, tr_y = (RangeTracker() for _ in range(4))
    with torch.no_grad():
        for i, (X, _) in enumerate(train_loader):
            if i >= n_calib_batches:
                break
            X = X.to(device)
            flat = features(X)                       # FC1 input
            h1 = torch.relu(fc1(flat))               # FC2 input  (post-relu -> fused)
            h2 = torch.relu(fc2(h1))                 # FC3 input  (post-relu -> fused)
            logits = fc3(h2)                         # output
            tr_x.update(flat); tr_h1.update(h1); tr_h2.update(h2); tr_y.update(logits)

    SX0, ZX0 = tr_x.qparams()
    S1,  Z1  = tr_h1.qparams()                       # fused ReLU boundary (Z1 = qmin)
    S2,  Z2  = tr_h2.qparams()                       # fused ReLU boundary (Z2 = qmin)
    SY,  ZY  = tr_y.qparams()

    # ============================== PREPARE (offline) ==============================
    # quantize weights ON CPU (integer matmul needs CPU); freeze all scales/zp
    cpu = lambda t: t.detach().cpu()
    qfc1 = QLinear.prepare(cpu(fc1.weight), cpu(fc1.bias), SX0, ZX0, S1, Z1)
    qfc2 = QLinear.prepare(cpu(fc2.weight), cpu(fc2.bias), S1,  Z1,  S2, Z2)
    qfc3 = QLinear.prepare(cpu(fc3.weight), cpu(fc3.bias), S2,  Z2,  SY, ZY)

    # ============================== SAVE the artifact ==============================
    artifact = {
        "qfc1": qfc1, "qfc2": qfc2, "qfc3": qfc3,    # int8 qW + int32 qbias + M0/shift + SY/ZY
        "SX0": SX0, "ZX0": ZX0,                       # classifier input calibration
        "conv": {k: v.cpu() for k, v in features.state_dict().items()},  # conv is still float
    }
    torch.save(artifact, save_path)

    # ---- report: show the classifier is integer-only ----
    print(f"saved -> {save_path}")
    for name, q in [("fc1", qfc1), ("fc2", qfc2), ("fc3", qfc3)]:
        print(f"  {name}: qW {tuple(q.qW.shape)} {q.qW.dtype}  "
              f"qbias {tuple(q.qbias.shape)} {q.qbias.dtype}  M0={q.M0} shift={q.shift}")
    print(f"  input calib: SX0={SX0:.6g}  ZX0={ZX0}")
    print(f"  file size: {os.path.getsize(save_path) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
