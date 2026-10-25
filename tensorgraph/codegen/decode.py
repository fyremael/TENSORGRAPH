"""Experimental typed multi-input DAG and CUDA lowering for a decode fragment.

This bounded IR does not participate in equality saturation yet. There is no
fallback to a handwritten candidate kernel or to another compiler.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class Node:
    op: str
    args: tuple[int, ...] = ()
    name: str = ""
    axis: str = "element"


@dataclass(frozen=True)
class DecodeGraph:
    dtype: str
    hidden: int
    epsilon: float = 1e-5

    def __post_init__(self):
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("unsupported dtype")
        if self.hidden not in (1024, 4096, 8192):
            raise ValueError("unsupported hidden extent")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive finite")

    @property
    def nodes(self):
        return (
            Node("input", name="x"), Node("input", name="residual"),
            Node("input", name="weight", axis="column"),
            Node("add", (0, 1)), Node("square", (3,)),
            Node("mean", (4,), axis="row"), Node("rsqrt_epsilon", (5,), axis="row"),
            Node("mul", (3, 6)), Node("mul", (7, 2)),
        )

    def contract(self):
        return dict(dtype=self.dtype, hidden=self.hidden, epsilon=self.epsilon,
                    nodes=[asdict(n) for n in self.nodes], outputs=[3, 8],
                    accumulation="float32", aliasing="all five tensors disjoint",
                    layout="contiguous", stream="current PyTorch CUDA stream",
                    nonfinite="IEEE propagation; overflow in square produces zero inverse RMS")

    def emit(self):
        types = {"float32": ("float", "Float", "v", "v"),
                 "float16": ("__half", "Half", "__half2float(v)", "__float2half_rn(v)"),
                 "bfloat16": ("__nv_bfloat16", "BFloat16", "__bfloat162float(v)", "__float2bfloat16_rn(v)")}
        storage, scalar, load, store = types[self.dtype]
        # Scalar statements are derived from the typed DAG, including the reduction input.
        lines = []
        for i, node in enumerate(self.nodes):
            args = [f"v{a}" for a in node.args]
            if node.op == "input":
                expr = f"read({node.name}[{'col' if node.axis == 'column' else 'idx'}])"
            elif node.op == "add": expr = " + ".join(args)
            elif node.op == "mul": expr = " * ".join(args)
            elif node.op == "square": expr = f"{args[0]} * {args[0]}"
            elif node.op in ("mean", "rsqrt_epsilon"): continue
            else: raise ValueError(f"unsupported node {node.op}")
            lines.append((i, f"float v{i} = {expr};"))
        before = "\n".join(s for i, s in lines if i <= 4)
        after = "\n".join(s for i, s in lines if i > 6)
        source = f'''#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
using S = {storage};
__device__ float read(S v) {{ return {load}; }}
__device__ S write(float v) {{ return {store}; }}
__global__ void decode(const S* x, const S* residual, const S* weight, S* rout, S* norm) {{
    constexpr int H = {self.hidden};
    __shared__ float sums[256];
    float sum = 0.f;
    for (int col=threadIdx.x; col<H; col+=256) {{
        int idx = blockIdx.x*H + col;
        {before}
        sum += v4;
    }}
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int stride=128; stride>0; stride/=2) {{
        if (threadIdx.x<stride) sums[threadIdx.x] += sums[threadIdx.x+stride];
        __syncthreads();
    }}
    float v5 = sums[0] / H;
    float v6 = rsqrtf(v5 + {self.epsilon:.17g}f);
    for (int col=threadIdx.x; col<H; col+=256) {{
        int idx = blockIdx.x*H + col;
        {before}
        {after}
        rout[idx] = write(v3);
        norm[idx] = write(v8);
    }}
}}
void run_out(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, torch::Tensor rout, torch::Tensor norm) {{
    TORCH_CHECK(x.is_cuda() && x.dim()==2 && x.size(0)>0 && x.size(1)=={self.hidden}, "invalid input");
    std::vector<torch::Tensor> tensors = {{x, residual, weight, rout, norm}};
    for (int i=0; i<5; ++i) {{
        const auto& t=tensors[i];
        TORCH_CHECK(t.is_cuda() && t.device()==x.device() && t.is_contiguous() && t.scalar_type()==at::ScalarType::{scalar}, "invalid tensor ABI");
        TORCH_CHECK(i==2 ? (t.dim()==1 && t.numel()=={self.hidden}) : t.sizes().equals(x.sizes()), "shape mismatch");
        for (int j=0; j<i; ++j) {{
            auto a=reinterpret_cast<uintptr_t>(t.data_ptr());
            auto b=reinterpret_cast<uintptr_t>(tensors[j].data_ptr());
            TORCH_CHECK(a+t.nbytes()<=b || b+tensors[j].nbytes()<=a, "overlapping tensor storage");
        }}
    }}
    c10::cuda::CUDAGuard guard(x.device());
    auto stream=at::cuda::getCurrentCUDAStream(x.get_device());
    decode<<<x.size(0),256,0,stream.stream()>>>((S*)x.data_ptr(),(S*)residual.data_ptr(),(S*)weight.data_ptr(),(S*)rout.data_ptr(),(S*)norm.data_ptr());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
'''
        return dict(contract=self.contract(), source=source,
                    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    graph_sha256=hashlib.sha256(json.dumps(self.contract(), sort_keys=True).encode()).hexdigest())


def compile_decode(artifact, build_directory):
    """Compile exactly the retained source, with no fast-math or implicit FMA."""
    source = artifact["source"]
    if hashlib.sha256(source.encode()).hexdigest() != artifact["source_sha256"]:
        raise ValueError("source hash mismatch")
    from torch.utils.cpp_extension import load_inline
    return load_inline(name="tg_decode_" + artifact["source_sha256"][:16],
                       cpp_sources="void run_out(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);",
                       cuda_sources=source, functions=["run_out"],
                       extra_cuda_cflags=["-O3", "--fmad=false"],
                       build_directory=str(build_directory), verbose=True)
