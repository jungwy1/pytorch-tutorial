"""
Apply the integer-only quantization (QTensor / QLinear) to AlexNet's CLASSIFIER.

The conv feature extractor stays in float; only the three FC layers are quantized:
    features(float) -> flat -> [ QLinear1 -> QLinear2 -> QLinear3 ] (int8) -> logits
FC1/FC2 use FUSED ReLU (calibrate on the POST-relu range -> ReLU folds into the
requant clamp, so no relu_int call is needed).

Calibration uses REAL Imagenette batches (not torch.randn): a few batches are run
through the float model to freeze the scale/zero-point at every FC boundary.

NOTE: torch integer matmul is not supported on CUDA, so the float conv runs on GPU
(fast) while the int8 classifier runs on CPU.
"""
import os
import torch

from model import AlexNet
from dataset import get_dataloaders
from quant_basics import QTensor, compute_qparams
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
    print(f"features on {device}, int8 classifier on cpu")

    # ---- paths: local vs Colab (mirror the pruning scripts) ----
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    if IN_COLAB:
        weight_path = "/content/drive/MyDrive/alexnet_pruned90.pth"
        path2data = "/content/data"
    else:
        weight_path = "../pruning/alexnet_pruned90.pth"
        path2data = "../data"

    # ---- load the trained float model ----
    model = AlexNet().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()                                     # dropout OFF

    train_loader, test_loader = get_dataloaders(batch_size=64, path2data=path2data)

    # ---- split: conv features (float) + the three FC layers ----
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

    def quantized_classifier(flat):
        # flat: (batch, in) float -> transpose to (in, batch) for qW @ qX
        qx = QTensor.quantize_with(cpu(flat).T, SX0, ZX0)   # frozen SX0, ZX0
        qh = qfc1.run(qx)                                    # fused ReLU
        qh = qfc2.run(qh)                                    # fused ReLU
        qy = qfc3.run(qh)
        return qy.dequantize().T                             # (batch, out) float

    # ================================ EVALUATE ================================
    # compare float top-1 vs int8-classifier top-1 on the val set
    n_eval_batches = 10                            # set an int to subsample (CPU int matmul is slow)
    correct_f = correct_q = total = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            if n_eval_batches is not None and i >= n_eval_batches:
                break
            X = X.to(device)
            flat = features(X)
            logits_f = fc3(torch.relu(fc2(torch.relu(fc1(flat))))).cpu()   # float reference
            logits_q = quantized_classifier(flat)                          # int8 path
            correct_f += (logits_f.argmax(1) == y).sum().item()
            correct_q += (logits_q.argmax(1) == y).sum().item()
            total += y.numel()

    print(f"float    top-1: {100 * correct_f / total:.2f}%   ({total} imgs)")
    print(f"int8 FC  top-1: {100 * correct_q / total:.2f}%")


if __name__ == "__main__":
    main()
