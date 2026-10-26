import torch
import triton
import triton.language as tl

@triton.jit
def fused_transformer_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x0 = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x0, mask=mask)