"""
Step 3: integer-only matrix multiply (Jacob et al. 2018).

Quantize A (activation, asymmetric) and B (weight, symmetric), run the matmul with
integer arithmetic only, then apply a single float scale (SA * SB) at the very end.

Key identity (symmetric weights => ZB = 0):
    sum_k (qA - ZA) * qB  =  qA @ qB  -  ZA * sum_k qB
                             (A) runtime    (C) precompute (weights only)
"""
import torch

from quant_basics import compute_qparams, quantize, dequantize, check


# Integer-only matmul: real (A @ B) computed via int8 quant + int32 accumulate.
def quantized_matmul(A, B, n_bits=8):
    # quantize inputs: activation asymmetric, weight symmetric (ZB = 0)
    SA, ZA = compute_qparams(A.min().item(), A.max().item(), n_bits)
    SB, ZB = compute_qparams(B.min().item(), B.max().item(), n_bits, symmetric=True)
    qA = quantize(A, SA, ZA, n_bits)
    qB = quantize(B, SB, ZB, n_bits, symmetric=True)

    # integer accumulate via the decomposition; ZB = 0 drops the other two terms
    acc = qA @ qB - ZA * qB.sum(dim=0)          # (A) runtime  -  (C) precompute

    # dequantize: combined scale SA*SB, no zero_point (weight is symmetric)
    return dequantize(acc, SA * SB, zero_point=0)


if __name__ == "__main__":
    torch.manual_seed(0)
    A = torch.randn(2, 4)          # activation (M, K)
    B = torch.randn(4, 3)          # weight     (K, N)
    C_ref = A @ B                  # float reference

    error = 0
    # (1) decomposition identity: direct form == precompute form (exact integers)
    SA, ZA = compute_qparams(A.min().item(), A.max().item(), 8)
    SB, ZB = compute_qparams(B.min().item(), B.max().item(), 8, symmetric=True)
    qA = quantize(A, SA, ZA, 8)
    qB = quantize(B, SB, ZB, 8, symmetric=True)
    direct     = (qA - ZA) @ qB
    decomposed = qA @ qB - ZA * qB.sum(dim=0)
    error += check("decomposition identity", (direct == decomposed).all().item())

    # (2) integer-only matmul reproduces the float result within quantization error
    C_hat = quantized_matmul(A, B, n_bits=8)
    rel_err = (C_hat - C_ref).abs().max().item() / C_ref.abs().max().item()
    print(f"max relative error = {rel_err:.4f}")
    error += check("int-only matmul ~ float (rel_err < 0.05)", rel_err < 0.05)

    print("\nmatmul test passed" if error == 0 else "\nmatmul test failed")
