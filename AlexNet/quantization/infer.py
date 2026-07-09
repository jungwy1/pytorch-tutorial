"""
INFERENCE-ONLY: load the int8 artifact produced by quantize.py and run AlexNet's
classifier with integer arithmetic only.

NO calibration, NO weight quantization here -- QLinear.prepare is never called. We only
load qW / qbias / M0 / shift (already integers) and call .run:

    float conv (loaded weights) -> flat -> quantize_with(SX0, ZX0)
        -> qfc1.run -> qfc2.run -> qfc3.run  (int8 logits) -> argmax

argmax runs directly on the int8 logits: real = SY*(q - ZY) is monotonic in q (SY > 0),
so the winning class is the same as on the dequantized logits -- no final dequant needed.
"""
import torch

from model import AlexNet
from dataset import get_dataloaders
from quant_basics import QTensor            # for quantize_with (and to unpickle QTensor)
from quant_linear_layer import QLinear      # needed so torch.load can unpickle the QLinear objects


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"features on {device}, int8 classifier on cpu")

    # ---- paths: local vs Colab ----
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    if IN_COLAB:
        artifact_path = "/content/drive/MyDrive/alexnet_int8_classifier.pth"
        path2data = "/content/data"
    else:
        artifact_path = "alexnet_int8_classifier.pth"
        path2data = "../data"

    # ---- load the int8 artifact (the deployable model) ----
    art = torch.load(artifact_path, map_location="cpu", weights_only=False)
    qfc1, qfc2, qfc3 = art["qfc1"], art["qfc2"], art["qfc3"]
    SX0, ZX0 = art["SX0"], art["ZX0"]

    # ---- rebuild ONLY the float conv feature extractor and load its weights ----
    model = AlexNet().to(device)
    features = model.model[:14]                      # conv/relu/pool ... Flatten
    features.load_state_dict(art["conv"])
    features.eval()

    _, test_loader = get_dataloaders(batch_size=64, path2data=path2data)

    # ---- integer-only classifier: flat -> int8 logits -> predicted class ----
    def classify(flat):
        qx = QTensor.quantize_with(flat.detach().cpu().T, SX0, ZX0)   # (in, batch), frozen SX0/ZX0
        qh = qfc1.run(qx)          # fused ReLU
        qh = qfc2.run(qh)          # fused ReLU
        qy = qfc3.run(qh)          # int8 logits (out, batch)
        return qy.q.argmax(0)      # argmax on int8 (monotonic) -> (batch,) predicted class

    # ---- evaluate ----
    n_eval_batches = None                            # set an int to subsample (CPU int matmul is slow)
    correct = total = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            if n_eval_batches is not None and i >= n_eval_batches:
                break
            X = X.to(device)
            flat = features(X)
            pred = classify(flat)                    # (batch,) on cpu
            correct += (pred == y).sum().item()
            total += y.numel()

    print(f"int8 classifier top-1: {100 * correct / total:.2f}%   ({total} imgs)")


if __name__ == "__main__":
    main()
