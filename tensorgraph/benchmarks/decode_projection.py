"""Persistent residual/RMSNorm + packed QKV projection experiment setup.

The native lane uses generated TENSORGRAPH normalization followed by PyTorch's
matrix multiply. It is a mixed backend pipeline, not an all-generated GEMM.
Allocation and execution validation are required before performance admission.
"""


def make_projection_lanes(torch, native, x, residual, weight, qkv_weight):
    """Allocate shared persistent outputs and baseline workspaces before timing.

    qkv_weight is [H, 3H], implementing equal-width Q/K/V projections (MHA).
    Normalization is rounded to the storage dtype before projection in every lane.
    The supplied native extension is compiled for this H/dtype by DecodeGraph.
    """
    batch, hidden = x.shape
    if qkv_weight.shape != (hidden, 3 * hidden):
        raise ValueError("packed QKV weight must have shape [H, 3H]")
    if residual.shape != x.shape or weight.shape != (hidden,):
        raise ValueError("normalization input shapes do not match")
    tensors = (x, residual, weight, qkv_weight)
    if any(t.device != x.device or t.dtype != x.dtype or not t.is_contiguous()
           for t in tensors):
        raise ValueError("inputs must share device/dtype and be contiguous")
    rout = torch.empty_like(x)
    normalized = torch.empty_like(x)
    qkv = torch.empty((batch, 3 * hidden), device=x.device, dtype=x.dtype)
    sx = torch.empty_like(x, dtype=torch.float32)
    sr = torch.empty_like(sx)
    ww = torch.empty_like(weight, dtype=torch.float32)
    square = torch.empty_like(sx)
    inverse = torch.empty((batch, 1), device=x.device, dtype=torch.float32)
    temporary = torch.empty_like(sx)

    def eager():
        sx.copy_(x)
        sr.copy_(residual)
        ww.copy_(weight)
        torch.add(sx, sr, out=sx)
        torch.mul(sx, sx, out=square)
        torch.mean(square, dim=-1, keepdim=True, out=inverse)
        inverse.add_(1e-5)
        torch.rsqrt(inverse, out=inverse)
        torch.mul(sx, inverse, out=temporary)
        torch.mul(temporary, ww, out=temporary)
        rout.copy_(sx)
        normalized.copy_(temporary)
        torch.mm(normalized, qkv_weight, out=qkv)

    def native_library():
        native.run_out(x, residual, weight, rout, normalized)
        torch.mm(normalized, qkv_weight, out=qkv)

    def compiled_function(x, residual, weight, qkv_weight, rout, normalized, qkv):
        s = x.float() + residual.float()
        n = s * torch.rsqrt((s * s).mean(-1, keepdim=True) + 1e-5) * weight.float()
        rout.copy_(s)
        normalized.copy_(n)
        # Reading the stored normalized tensor makes rounding an explicit boundary.
        torch.mm(normalized, qkv_weight, out=qkv)

    compiled = torch.compile(compiled_function, fullgraph=True)

    def inductor():
        compiled(x, residual, weight, qkv_weight, rout, normalized, qkv)

    return {
        "outputs": (rout, normalized, qkv),
        "workspaces": (sx, sr, ww, square, inverse, temporary),
        "lanes": {"eager_library": eager, "native_library": native_library,
                  "inductor": inductor},
        "scope": "residual + weighted RMSNorm + packed MHA QKV projection",
        "native_projection_backend": "PyTorch matrix multiplication; not generated native GEMM",
    }
