"""
Step 5: chain two integer-only Linear layers with a FUSED ReLU (Jacob et al. 2018).

Float reference:
    pre = W1 X + b1 ;  H = relu(pre) ;  Y = W2 H + b2
    (pre = pre-activation, a REAL value -- not to be confused with a zero-point Z*)

Integer-only path -- NO dequantize in the middle, and NO explicit ReLU op:
    qx = quantize_with(X, SX, ZX)           # only the very first input
    qh = layer1.run(qx)                      # int8 QTensor (carries SH, ZH)
    qy = layer2.run(qh)                      # int8 QTensor (carries SY, ZY)
    Y  = qy.dequantize()                     # dequantize only at the very end

Two ideas make this work:
  1. layer1's OUTPUT calibration (SH, ZH) IS layer2's INPUT calibration, so the int8
     tensor flows straight from one layer into the next -- that is the integer-only chain.
  2. FUSED ReLU: we calibrate (SH, ZH) on the POST-relu activation H (range [0, max]).
     r_min = 0 forces ZH = qmin, and q = ZH means real 0. layer1's requant already clamps
     its output to [qmin, qmax]; with ZH = qmin that lower clamp maps every negative
     pre-activation to real 0 -- the ReLU is absorbed into the requant clamp for free.
     (relu_int below is NOT needed here; kept only for the non-fused variant.)
"""
import torch

from quant_basics import QTensor, compute_qparams, check
from quant_linear_layer import QLinear


def relu_int(x: QTensor) -> QTensor:
    # NOT used in the fused path (requant clamp already does ReLU when ZH = qmin).
    # Kept for the non-fused variant (SH,ZH calibrated on the pre-activation, ZH interior):
    #   real = S*(q - Z) >= 0  <=>  q >= Z,  so ReLU = clamp the integer floor up to Z.
    q = torch.clamp(x.q, min=x.zero_point).to(torch.int8)
    return QTensor(q, x.scale, x.zero_point)


if __name__ == "__main__":
    torch.manual_seed(0)
    IN, HID, OUT = 4, 5, 3
    W1 = torch.randn(HID, IN);  b1 = torch.randn(HID)     # layer 1  (hid, in)
    W2 = torch.randn(OUT, HID); b2 = torch.randn(OUT)     # layer 2  (out, hid)

    # ---- calibration (offline): FREEZE the scale/zp at every point from a batch ----
    X_calib  = torch.randn(IN, 256)
    pre_calib = W1 @ X_calib + b1.unsqueeze(1)            # pre-activation of layer 1 (relu 전)
    H_calib  = torch.relu(pre_calib)                     # post-activation feeding layer 2
    Y_calib  = W2 @ H_calib + b2.unsqueeze(1)

    SX, ZX = compute_qparams(X_calib.min().item(), X_calib.max().item(), n_bits=8)
    # FUSED ReLU: calibrate on POST-relu H (min = 0) -> ZH = qmin, ReLU folds into requant
    SH, ZH = compute_qparams(H_calib.min().item(), H_calib.max().item(), n_bits=8)
    SY, ZY = compute_qparams(Y_calib.min().item(), Y_calib.max().item(), n_bits=8)

    # ---- offline: prepare both layers.  layer1's output (SH,ZH) = layer2's input ----
    layer1 = QLinear.prepare(W1, b1, SX, ZX, SH, ZH)
    layer2 = QLinear.prepare(W2, b2, SH, ZH, SY, ZY)

    # ---- inference (online): a NEW input, integer-only through the whole chain ----
    X_new = torch.randn(IN, 3)
    Y_ref = W2 @ torch.relu(W1 @ X_new + b1.unsqueeze(1)) + b2.unsqueeze(1)

    qx = QTensor.quantize_with(X_new, SX, ZX)      # frozen SX, ZX
    qh = layer1.run(qx)                            # int8 -> int8  (ReLU folded into clamp)
    qy = layer2.run(qh)                            # int8 -> int8
    Y_hat = qy.dequantize()                        # dequantize only here

    rel_err = (Y_hat - Y_ref).abs().max().item() / Y_ref.abs().max().item()
    print(f"max relative error = {rel_err:.4f}")
    error = check("int-only 2-layer chain (fused ReLU) ~ float (rel_err < 0.06)", rel_err < 0.06)
    print("\nchain test passed" if error == 0 else "\nchain test failed")
