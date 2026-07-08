"""
Affine quantization primitives (Jacob et al. 2018 scheme).

    real = scale * (q - zero_point)          # dequantize
    q    = round(real / scale) + zero_point  # quantize (then clamp)

asymmetric (zero_point != 0) -> activations;  symmetric (zero_point = 0) -> weights.
"""
import torch
from dataclasses import dataclass

# Integer grid [qmin, qmax] for a given bit width.
#   symmetric  -> [-127, 127]  (drop -128 so the range is centered on 0)
#   asymmetric -> [-128, 127]  (full signed range; zero_point handles the offset)
def qrange(n_bits, symmetric = False):
    qmax = 2 ** (n_bits - 1) - 1
    qmin = -qmax if symmetric else (-qmax - 1)
    return qmin, qmax

# Pick (scale, zero_point) that map the real range [r_min, r_max] onto the grid.
def compute_qparams(r_min, r_max, n_bits, symmetric = False):
    qmin, qmax = qrange(n_bits=n_bits, symmetric=symmetric)

    if not symmetric: # Asymmetric: fit [r_min, r_max] exactly, offset via zero_point
        scale = (r_max - r_min) / (qmax - qmin)
        # zero_point = integer that real 0.0 maps to; clamp to stay on the grid
        zero_point = round(qmin - r_min / scale)
        zero_point = max(min(zero_point, qmax), qmin)
    else: # Symmetric: range centered on 0, so zero_point is fixed at 0
        scale = max(abs(r_min), abs(r_max)) / qmax
        zero_point = 0

    return scale, zero_point

# Real tensor -> integer tensor:  q = clamp(round(r/scale) + zero_point, qmin, qmax)
def quantize(r, scale, zero_point, n_bits, symmetric = False):
    qmin, qmax = qrange(n_bits=n_bits, symmetric=symmetric)
    q = torch.round(r / scale) + zero_point
    q = torch.clamp(q, qmin, qmax)
    return q.to(torch.int32)             # generic int container; QTensor casts to int8/int32

# Integer tensor -> approximate real tensor (shared by both flavors; sym passes zp=0).
def dequantize(q, scale, zero_point):
    r_hat = scale * (q.to(torch.int32) - zero_point)
    return r_hat


# A quantized value is (q, scale, zero_point) as one bundle: the integer alone is
# meaningless, you always need scale/zero_point to recover real = scale*(q - zp).
@dataclass
class QTensor:
    q: torch.Tensor      # integer tensor (int8 for weight/activation, int32 for bias/acc)
    scale: float         # S
    zero_point: int      # Z  (0 for symmetric)

    def dequantize(self):
        # widen to int32 first so int8 - zero_point does not wrap
        return self.scale * (self.q.to(torch.int32) - self.zero_point)

    @classmethod
    def from_float(cls, r, symmetric=False):
        # weight / calibration: DECIDE scale/zp from the data's own min/max
        scale, zp = compute_qparams(r.min().item(), r.max().item(), 8, symmetric)
        q = quantize(r, scale, zp, 8, symmetric).to(torch.int8)
        return cls(q, scale, zp)

    @classmethod
    def quantize_with(cls, r, scale, zero_point, symmetric=False):
        # runtime activation: quantize with an ALREADY-frozen scale/zp (from calibration),
        # NOT recomputed from this input's own range.
        q = quantize(r, scale, zero_point, 8, symmetric).to(torch.int8)
        return cls(q, scale, zero_point)

    @classmethod
    def quantize_bias(cls, bias, scale_x, qW):
        # bias scale = S_X * S_W (accumulator scale), symmetric (zp=0), stored int32.
        # takes the activation SCALE (a calibration constant, known offline), not a
        # runtime QTensor -- the bias is prepared before any input exists.
        scale = scale_x * qW.scale
        q = torch.round(bias / scale).to(torch.int32)
        return cls(q, scale, 0)


# Validation
# Tiny test helper: print PASS/FAIL and return 0/1 so callers can sum failures.
def check(name, condition):
    if condition:
        print(f"[PASS] {name}")
        return 0
    else:
        print(f"[FAIL] {name}")
        return 1

if __name__ == "__main__":
    # ---------------------------------- qrange ----------------------------------
    print("======================= qrange test ========================")
    error = 0
    error += check("8-bit asymmetric quant range test", qrange(8) == (-128, 127))
    error += check("8-bit symmetric quant range test", qrange(8, symmetric=True) == (-127, 127))
    error += check("4-bit asymmetric quant range test", qrange(4) == (-8, 7))

    print("\nqrange test passed") if (error == 0) else print("\nqrange test failed")
    # print("============================================================")
    # ---------------------------------- qparams ----------------------------------
    print("======================= qparams test ========================")
    print("===================== 8-bit asymmetric ======================")
    qmin, qmax = qrange(8)

    error = 0
    # case 1: typical range including zero
    print("case 1: typical range including zero")
    r_min, r_max = -1.0, 3.0
    scale, zp = compute_qparams(r_min, r_max, n_bits=8)
    print(f"[-1, 3]  scale={scale:.6f}  zero_point={zp}")
    error += check("scale > 0 test", scale > 0)
    error += check("zero point out of grid test\n", qmin <= zp <= qmax)

    # case 2: range excluding zero
    print("case 2: range excluding zero")
    r_min, r_max = 2.0, 5.0
    scale, zp = compute_qparams(r_min, r_max, n_bits=8)
    print(f"[2, 5]  scale={scale:.6f}  zero_point={zp}")
    error += check("scale > 0 test", scale > 0)
    error += check("zero point out of grid test\n", qmin <= zp <= qmax)

    print("\nqparams test passed") if (error == 0) else print("\nqparams test failed")
    # ---------------------------- quantize / dequantize ----------------------------
    print("\n======================= quantize test ========================")
    error = 0

    # (a) asymmetric endpoint: r_min -> qmin, r_max -> qmax  (실제 quantize 호출)
    qmin, qmax = qrange(8)
    r = torch.tensor([-1.0, 3.0])
    scale, zp = compute_qparams(-1.0, 3.0, n_bits=8)
    q = quantize(r, scale, zp, n_bits=8)
    error += check("asym r_min -> qmin", q[0].item() == qmin)
    error += check("asym r_max -> qmax", q[1].item() == qmax)

    # (b) asymmetric round-trip: dequantize(quantize(r)) ~ r,  오차 <= scale
    r = torch.randn(1000) * 3.0
    scale, zp = compute_qparams(r.min().item(), r.max().item(), n_bits=8)
    r_hat = dequantize(quantize(r, scale, zp, n_bits=8), scale, zp)
    error += check("asym roundtrip err <= scale", (r - r_hat).abs().max().item() <= scale)

    # (c) symmetric round-trip
    w = torch.randn(1000) * 0.5
    scale, zp = compute_qparams(w.min().item(), w.max().item(), n_bits=8, symmetric=True)
    w_hat = dequantize(quantize(w, scale, zp, n_bits=8, symmetric=True), scale, zp)
    error += check("sym roundtrip err <= scale", (w - w_hat).abs().max().item() <= scale)

    # (d) symmetric maps 0.0 -> integer 0 (그 precompute 잘 되는 성질)
    z = quantize(torch.tensor([0.0]), scale, 0, n_bits=8, symmetric=True)
    error += check("sym 0.0 -> 0", z.item() == 0)

    print("\nquantize test passed") if (error == 0) else print("\nquantize test failed")

    # ---------------------------------- QTensor ----------------------------------
    print("\n======================= QTensor test ========================")
    error = 0

    # (a) from_float: weight (symmetric) -> int8 storage, roundtrip err <= scale
    w = torch.randn(1000) * 0.5
    qw = QTensor.from_float(w, symmetric=True)
    error += check("from_float stores int8", qw.q.dtype == torch.int8)
    error += check("from_float roundtrip err <= scale",
                   (w - qw.dequantize()).abs().max().item() <= qw.scale)

    # (b) from_float: activation (asymmetric) -> int8 storage
    x = torch.randn(1000) * 3.0
    qx = QTensor.from_float(x)
    error += check("from_float asym stores int8", qx.q.dtype == torch.int8)
    error += check("from_float asym roundtrip err <= scale",
                   (x - qx.dequantize()).abs().max().item() <= qx.scale)

    # (c) quantize_bias: scale = S_X * S_W, zp = 0, stored int32 (wide, no int8 clamp)
    b = torch.randn(5)
    qb = QTensor.quantize_bias(b, qx.scale, qw)
    error += check("quantize_bias stores int32", qb.q.dtype == torch.int32)
    error += check("quantize_bias scale == SX*SW", abs(qb.scale - qx.scale * qw.scale) < 1e-12)
    error += check("quantize_bias err <= scale",
                   (b - qb.dequantize()).abs().max().item() <= qb.scale)

    # (d) int8 dequantize must NOT wrap:  1.0 * (127 - (-57)) = 184, not -72
    qt = QTensor(torch.tensor([127], dtype=torch.int8), 1.0, -57)
    error += check("int8 dequantize no wrap", qt.dequantize().item() == 184)

    print("\nQTensor test passed") if (error == 0) else print("\nQTensor test failed")


