"""
Integer-only Linear layer  Y = W X + b  (Jacob et al. 2018), split into an offline
PREPARE step and an online RUN step, all built on QTensor.

  prepare : quantize W/b once, fold the -Z_X*sum(qW) correction and the fixed-point
            multiplier (M0, shift) into stored integers  ->  what a deployed model holds.
  run     : take an int8 QTensor input, compute qW@qX + q_bias (int32 accumulate),
            requantize to an int8 QTensor output.

Slide formulation (Z_W = Z_b = 0,  S_b = S_W S_X):
  q_Y = M * (qW qX + q_bias) + Z_Y,   q_bias = q_b - Z_X * sum_k qW,   M = S_W S_X / S_Y
  M is turned into fixed-point (M0, shift) so the requant stays integer-only.
"""
import torch
from dataclasses import dataclass

from quant_basics import QTensor, compute_qparams, qrange, check


@dataclass
class QLinear:
    qW: torch.Tensor      # int8 weight (out, in)
    qbias: torch.Tensor   # int32, folded:  q_b - Z_X * sum_k qW
    M0: int               # fixed-point multiplier (~ Q0.31)
    shift: int            # right-shift amount (= n + 31)
    SY: float             # output scale
    ZY: int               # output zero-point

    # ---------- offline: quantize weights once, precompute the constants ----------
    @classmethod
    def prepare(cls, W, b, SX, ZX, SY, ZY):
        qW = QTensor.from_float(W, symmetric=True)          # int8, zp = 0
        qb = QTensor.quantize_bias(b, SX, qW)               # int32, scale = SX * SW

        # fold the input zero-point correction into the bias (all known offline)
        qbias = qb.q - ZX * qW.q.sum(dim=1).to(torch.int32)

        # M = S_W S_X / S_Y  ->  fixed-point (M0, shift):  M ~= M0 * 2^-(31 + n)
        M = qW.scale * SX / SY
        n = 0
        while M < 0.5:
            M *= 2
            n += 1
        M0 = round(M * (2 ** 31))
        shift = n + 31

        return cls(qW.q, qbias, M0, shift, SY, ZY)

    # ---------- online: int8 input -> int8 output, integer arithmetic only ----------
    def run(self, x: QTensor) -> QTensor:
        # int8 operands, int32 accumulator; bias -> (out,1) to broadcast over batch
        acc = self.qW.to(torch.int32) @ x.q.to(torch.int32) + self.qbias.unsqueeze(1)

        # requant: multiply in int64, rounding right shift, then add output zero-point
        qY = (acc.to(torch.int64) * self.M0 + (1 << (self.shift - 1))) >> self.shift
        qY = qY + self.ZY

        qmin, qmax = qrange(n_bits=8)
        qY = torch.clamp(qY, qmin, qmax).to(torch.int8)
        return QTensor(qY, self.SY, self.ZY)             # int8 QTensor for the next layer


if __name__ == "__main__":
    torch.manual_seed(0)
    W = torch.randn(2, 4)                     # weight  (out, in)
    b = torch.randn(2,)                       # bias    (out,)

    # ---- calibration (offline): FREEZE activation scales from a representative batch ----
    X_calib = torch.randn(4, 256)             # many samples (in, batch)
    Y_calib = W @ X_calib + b.unsqueeze(1)
    SX, ZX = compute_qparams(X_calib.min().item(), X_calib.max().item(), n_bits=8)
    SY, ZY = compute_qparams(Y_calib.min().item(), Y_calib.max().item(), n_bits=8)

    # ---- offline: build the deployed layer with the FROZEN params ----
    layer = QLinear.prepare(W, b, SX, ZX, SY, ZY)

    # ---- inference (online): a NEW input, quantized with the FROZEN SX,ZX (not recomputed) ----
    X_new = torch.randn(4, 3)
    Y_ref = W @ X_new + b.unsqueeze(1)                 # float reference for the new input
    qx = QTensor.quantize_with(X_new, SX, ZX)         # frozen scale/zp, NOT from X_new's range
    qy = layer.run(qx)
    Y_hat = qy.dequantize()

    rel_err = (Y_hat - Y_ref).abs().max().item() / Y_ref.abs().max().item()
    print(f"max relative error = {rel_err:.4f}")
    error = check("int-only linear ~ float (rel_err < 0.05)", rel_err < 0.05)
    print("\nlinear test passed" if error == 0 else "\nlinear test failed")
