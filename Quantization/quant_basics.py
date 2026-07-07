"""
Affine quantization primitives (Jacob et al. 2018 scheme).

    real = scale * (q - zero_point)          # dequantize
    q    = round(real / scale) + zero_point  # quantize (then clamp)

asymmetric (zero_point != 0) -> activations;  symmetric (zero_point = 0) -> weights.
"""
import torch

# Integer grid [qmin, qmax] for a given bit width.
#   symmetric  -> [-127, 127]  (drop -128 so the range is centered on 0)
#   asymmetric -> [-128, 127]  (full signed range; zero_point handles the offset)
def qrange(n_bits, symmetric = False):
    qmax = 2 ** (n_bits - 1) - 1
    qmin = -qmax if symmetric else (-qmax - 1)
    return qmin, qmax

# Pick (scale, zero_point) that map the real range [r_min, r_max] onto the grid.
def compute_qparams(r_min, r_max, n_bits, symmetric = False):
    qmin, qmax = qrange(n_bits, symmetric=symmetric)

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
    qmin, qmax = qrange(n_bits, symmetric=symmetric)
    q = torch.round(r / scale) + zero_point
    q = torch.clamp(q, qmin, qmax)
    return q.to(torch.int32)              # cast so the result is truly integer

# Integer tensor -> approximate real tensor (shared by both flavors; sym passes zp=0).
def dequantize(q, scale, zero_point):
    r_hat = scale * (q - zero_point)
    return r_hat


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


