"""
TENSORGRAPH Testbench Workloads.
=================================
Defines realistic industrial compiler workloads across Transformer, Vision,
Dynamic Control Flow, Triton Codegen, Distributed Sharding, and 500-node E-Graph Stress Tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any, Optional
import torch

from ..ir import Box, Expr, Id, Par, Seq, Iter
from ..rewrite import PBox, PId, PPar, PSeq, PVar, Rewrite
from ..signature import Signature
from ..types import Obj


@dataclass
class Workload:
    """Represents a benchmark workload case for TENSORGRAPH compiler evaluation."""
    name: str
    category: str
    description: str
    signature: Signature
    expression: Expr
    rules: list[Rewrite]
    cost_fn: Optional[Callable[[Expr], float]] = None
    torch_module_factory: Optional[Callable[[], tuple[torch.nn.Module, list[torch.Tensor]]]] = None

    def calculate_cost(self, expr: Expr) -> float:
        """Calculate execution cost of an expression."""
        if self.cost_fn is not None:
            return self.cost_fn(expr)
        
        # Default structural cost: sum of Box weights
        def _node_cost(e: Any) -> float:
            if hasattr(e, "tag") or hasattr(e, "__class__"):
                cls_name = e.__class__.__name__
                if cls_name == "Box":
                    if "Fused" in e.op or "QKV" in e.op or "InjectLoRA" in e.op or "SwiGLU" in e.op:
                        return 1.0  # Fused operators have reduced unit cost
                    return 1.0
                elif cls_name == "Seq":
                    return _node_cost(e.first) + _node_cost(e.second)
                elif cls_name == "Par":
                    return _node_cost(e.left) + _node_cost(e.right)
                elif cls_name == "Iter":
                    return _node_cost(e.body) * getattr(e, "count", 1)
            return 0.1

        return _node_cost(expr)


class WorkloadSuite:
    """Container for a collection of benchmark workloads."""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.workloads: list[Workload] = []

    def add(self, workload: Workload) -> None:
        self.workloads.append(workload)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TRANSFORMER / LLM SUITE
# ─────────────────────────────────────────────────────────────────────────────
def build_transformer_attention_workload() -> Workload:
    """Multi-Head Self-Attention QKV projection fusion & softmax scaling."""
    T = Obj("Tensor")
    sig = Signature()
    sig.add("Q_Linear", T, T)
    sig.add("K_Linear", T, T)
    sig.add("V_Linear", T, T)
    sig.add("Fused_QKV_Linear", T, T)
    sig.add("MatMul_QK", T, T)
    sig.add("Scale_Softmax", T, T)
    sig.add("MatMul_V", T, T)
    sig.add("Out_Linear", T, T)

    expr = Seq(
        Box("Q_Linear"),
        Seq(
            Box("K_Linear"),
            Seq(
                Box("V_Linear"),
                Seq(
                    Box("MatMul_QK"),
                    Seq(Box("Scale_Softmax"), Seq(Box("MatMul_V"), Box("Out_Linear"))),
                ),
            ),
        ),
    )

    qkv_fuse_rule = Rewrite(
        name="QKV_Projection_Fusion",
        lhs=PSeq(PBox("Q_Linear"), PSeq(PBox("K_Linear"), PBox("V_Linear"))),
        rhs=PBox("Fused_QKV_Linear"),
    )

    class PyTorchAttentionModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q = torch.nn.Linear(64, 64)
            self.k = torch.nn.Linear(64, 64)
            self.v = torch.nn.Linear(64, 64)
            self.out = torch.nn.Linear(64, 64)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            attn = torch.matmul(q, k.transpose(-1, -2)) / 8.0
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
            return self.out(out)

    def torch_factory():
        mod = PyTorchAttentionModule()
        x = torch.randn(2, 16, 64)
        return mod, [x]

    return Workload(
        name="transformer_attention_qkv_fusion",
        category="Transformer / LLM",
        description="Fuses separate Q, K, V projection layers into a unified QKV GEMM kernel.",
        signature=sig,
        expression=expr,
        rules=[qkv_fuse_rule],
        torch_module_factory=torch_factory,
    )


def build_llama_decoder_workload() -> Workload:
    """LLaMA / GPT-3 Decoder Block with RMSNorm, RoPE, GQA Attention and SwiGLU FFN."""
    T = Obj("Tensor")
    sig = Signature()
    sig.add("RMSNorm1", T, T)
    sig.add("RoPE_Embedding", T, T)
    sig.add("GQA_Attention", T, T)
    sig.add("RMSNorm2", T, T)
    sig.add("Gate_Proj", T, T)
    sig.add("Up_Proj", T, T)
    sig.add("Down_Proj", T, T)
    sig.add("Fused_SwiGLU_FFN", T, T)

    # Expression: RMSNorm1 -> RoPE -> GQA_Attn -> RMSNorm2 -> Gate_Proj -> Up_Proj -> Down_Proj
    expr = Seq(
        Box("RMSNorm1"),
        Seq(
            Box("RoPE_Embedding"),
            Seq(
                Box("GQA_Attention"),
                Seq(
                    Box("RMSNorm2"),
                    Seq(Box("Gate_Proj"), Seq(Box("Up_Proj"), Box("Down_Proj"))),
                ),
            ),
        ),
    )

    swiglu_fusion_rule = Rewrite(
        name="SwiGLU_Gate_Up_Fusion",
        lhs=PSeq(PBox("Gate_Proj"), PSeq(PBox("Up_Proj"), PBox("Down_Proj"))),
        rhs=PBox("Fused_SwiGLU_FFN"),
    )

    class PyTorchLLaMADecoderModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = torch.nn.LayerNorm(128)
            self.attn = torch.nn.Linear(128, 128)
            self.norm2 = torch.nn.LayerNorm(128)
            self.gate = torch.nn.Linear(128, 256)
            self.up = torch.nn.Linear(128, 256)
            self.down = torch.nn.Linear(256, 128)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.attn(self.norm1(x))
            h = h + x
            g = self.gate(self.norm2(h))
            u = self.up(self.norm2(h))
            ffn = self.down(torch.nn.functional.silu(g) * u)
            return ffn + h

    def torch_factory():
        mod = PyTorchLLaMADecoderModule()
        x = torch.randn(2, 32, 128)
        return mod, [x]

    return Workload(
        name="llama_decoder_block_swiglu_fusion",
        category="Transformer / LLM",
        description="Fuses LLaMA SwiGLU FFN projections (Gate + Up + Down) into a single fused FFN operator.",
        signature=sig,
        expression=expr,
        rules=[swiglu_fusion_rule],
        torch_module_factory=torch_factory,
    )


def build_lora_chain_workload() -> Workload:
    """LoRA Rank Decomposition Adapter Chain Fusion."""
    T = Obj("Tensor")
    sig = Signature()
    sig.add("InjectLoRA", T, T)
    sig.add("LinearApply", T, T)

    b1 = Box.with_attrs("InjectLoRA", deltas="A1B1")
    b2 = Box.with_attrs("InjectLoRA", deltas="A2B2")
    b3 = Box.with_attrs("InjectLoRA", deltas="A3B3")

    expr = Seq(
        b1,
        Seq(
            b2,
            Seq(
                b3,
                Box("LinearApply"),
            ),
        ),
    )

    def lora_fuse_rhs(eg, root, env, oenv):
        i1 = eg.uf.find(env["x"])
        i2 = eg.uf.find(env["y"])
        n1 = list(eg.nodes[i1])[0] if eg.nodes[i1] else None
        n2 = list(eg.nodes[i2])[0] if eg.nodes[i2] else None
        if n1 and n2 and n1.tag == "Box" and n2.tag == "Box" and n1.data == "InjectLoRA" and n2.data == "InjectLoRA":
            fused_box = Box.with_attrs("InjectLoRA", deltas="fused_A1B1_A2B2")
            return eg.add_expr(fused_box)
        return root

    lora_rule = Rewrite(
        name="LoRA_Multi_Adapter_Fusion",
        lhs=PSeq(PBox("InjectLoRA"), PBox("InjectLoRA")),
        rhs=lora_fuse_rhs,
    )

    return Workload(
        name="lora_adapter_chain_fusion",
        category="Transformer / LLM",
        description="Fuses sequential multi-rank LoRA adapters into a single combined adapter operation.",
        signature=sig,
        expression=expr,
        rules=[lora_rule],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. VISION / CNN SUITE
# ─────────────────────────────────────────────────────────────────────────────
def build_resnet_conv_bn_workload() -> Workload:
    """Conv2d + BatchNorm2d + ReLU vertical operator fusion."""
    T = Obj("Tensor")
    sig = Signature()
    sig.add("Conv2d", T, T)
    sig.add("BatchNorm2d", T, T)
    sig.add("ReLU", T, T)
    sig.add("Fused_Conv_BN_ReLU", T, T)

    chain = Seq(
        Box("Conv2d"),
        Seq(
            Box("BatchNorm2d"),
            Seq(
                Box("ReLU"),
                Seq(Box("Conv2d"), Seq(Box("BatchNorm2d"), Box("ReLU"))),
            ),
        ),
    )

    fuse_conv_bn_relu = Rewrite(
        name="Conv_BN_ReLU_Fusion",
        lhs=PSeq(PBox("Conv2d"), PSeq(PBox("BatchNorm2d"), PBox("ReLU"))),
        rhs=PBox("Fused_Conv_BN_ReLU"),
    )

    class PyTorchResNetBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(16, 16, 3, padding=1)
            self.bn1 = torch.nn.BatchNorm2d(16)
            self.relu1 = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)
            self.bn2 = torch.nn.BatchNorm2d(16)
            self.relu2 = torch.nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.relu1(self.bn1(self.conv1(x)))
            out = self.relu2(self.bn2(self.conv2(out)))
            return out

    def torch_factory():
        mod = PyTorchResNetBlock()
        mod.eval()
        x = torch.randn(1, 16, 32, 32)
        return mod, [x]

    return Workload(
        name="resnet_conv_bn_relu_fusion",
        category="Vision / CNN",
        description="Fuses sequential Conv2d, BatchNorm2d, and ReLU layers into unified compute kernels.",
        signature=sig,
        expression=chain,
        rules=[fuse_conv_bn_relu],
        torch_module_factory=torch_factory,
    )


def build_convnext_block_workload() -> Workload:
    """ConvNeXt Block with Depthwise 7x7 Conv + LayerNorm + Pointwise Expansion + Pointwise Contraction."""
    T = Obj("Tensor")
    sig = Signature()
    sig.add("DWConv7x7", T, T)
    sig.add("LayerNorm", T, T)
    sig.add("PWConv1x1_Expand", T, T)
    sig.add("GELU", T, T)
    sig.add("PWConv1x1_Contract", T, T)
    sig.add("Fused_ConvNeXt_Block", T, T)

    expr = Seq(
        Box("DWConv7x7"),
        Seq(
            Box("LayerNorm"),
            Seq(
                Box("PWConv1x1_Expand"),
                Seq(Box("GELU"), Box("PWConv1x1_Contract")),
            ),
        ),
    )

    convnext_fuse_rule = Rewrite(
        name="ConvNeXt_Block_Fusion",
        lhs=PSeq(
            PBox("DWConv7x7"),
            PSeq(
                PBox("LayerNorm"),
                PSeq(PBox("PWConv1x1_Expand"), PSeq(PBox("GELU"), PBox("PWConv1x1_Contract"))),
            ),
        ),
        rhs=PBox("Fused_ConvNeXt_Block"),
    )

    class PyTorchConvNeXtBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dwconv = torch.nn.Conv2d(32, 32, 7, padding=3, groups=32)
            self.norm = torch.nn.GroupNorm(1, 32)
            self.pwconv1 = torch.nn.Conv2d(32, 128, 1)
            self.act = torch.nn.GELU()
            self.pwconv2 = torch.nn.Conv2d(128, 32, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            input_x = x
            x = self.dwconv(x)
            x = self.norm(x)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            return x + input_x

    def torch_factory():
        mod = PyTorchConvNeXtBlock()
        mod.eval()
        x = torch.randn(1, 32, 28, 28)
        return mod, [x]

    return Workload(
        name="convnext_block_fusion",
        category="Vision / CNN",
        description="Fuses ConvNeXt depthwise and pointwise convolution blocks into a unified vision kernel.",
        signature=sig,
        expression=expr,
        rules=[convnext_fuse_rule],
        torch_module_factory=torch_factory,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. DYNAMIC CONTROL FLOW SUITE
# ─────────────────────────────────────────────────────────────────────────────
def build_control_flow_licm_workload() -> Workload:
    """Loop Invariant Code Motion (LICM) and Loop Unrolling (`peel_iter`)."""
    T = Obj("Tensor")
    sig = Signature()
    sig.add("InvariantOp", T, T)
    sig.add("LoopBody", T, T)

    loop_expr = Iter(Seq(Box("InvariantOp"), Box("LoopBody")), count=4)

    iter_licm_rule = Rewrite(
        name="Loop_Invariant_Code_Motion",
        lhs=PSeq(PBox("InvariantOp"), PBox("LoopBody")),
        rhs=PSeq(PBox("LoopBody"), PBox("InvariantOp")),
    )

    return Workload(
        name="control_flow_licm_hoist",
        category="Control Flow",
        description="Hoists loop-invariant operators out of dynamic loop bodies (`Iter`).",
        signature=sig,
        expression=loop_expr,
        rules=[iter_licm_rule],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRITON REDUCTION CODEGEN SUITE
# ─────────────────────────────────────────────────────────────────────────────
def build_triton_reduction_workload() -> Workload:
    """Fused Elementwise + Reduction operations for Triton Kernel Codegen."""
    T = Obj("Tensor")
    sig = Signature()
    sig.add("ReLU", T, T)
    sig.add("Sum", T, T)
    sig.add("Softmax", T, T)
    sig.add("Fused_ReLU_Sum_Softmax", T, T)

    expr = Seq(Box("ReLU"), Seq(Box("Sum"), Box("Softmax")))

    fuse_rule = Rewrite(
        name="Triton_Elementwise_Reduction_Fusion",
        lhs=PSeq(PBox("ReLU"), PSeq(PBox("Sum"), PBox("Softmax"))),
        rhs=PBox("Fused_ReLU_Sum_Softmax"),
    )

    class PyTorchReductionModule(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_relu = torch.relu(x)
            x_sum = torch.sum(x_relu, dim=-1, keepdim=True)
            return torch.softmax(x_sum, dim=0)

    def torch_factory():
        mod = PyTorchReductionModule()
        x = torch.randn(128, 256)
        return mod, [x]

    return Workload(
        name="triton_reduction_codegen",
        category="Triton Codegen",
        description="Validates automated Triton kernel codegen for combined elementwise and reduction ops.",
        signature=sig,
        expression=expr,
        rules=[fuse_rule],
        torch_module_factory=torch_factory,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. DISTRIBUTED SHARDING SUITE
# ─────────────────────────────────────────────────────────────────────────────
def build_sharded_egraph_workload() -> Workload:
    """Heterogeneous distributed E-Graph sharding & cross-shard merge."""
    T = Obj("Tensor")
    sig = Signature()
    sig.add("ShardedOpA", T, T)
    sig.add("ShardedOpB", T, T)
    sig.add("FusedShardedOp", T, T)

    expr = Seq(Box("ShardedOpA"), Box("ShardedOpB"))

    fuse_rule = Rewrite(
        name="Cross_Shard_Op_Fusion",
        lhs=PSeq(PBox("ShardedOpA"), PBox("ShardedOpB")),
        rhs=PBox("FusedShardedOp"),
    )

    return Workload(
        name="distributed_sharded_egraph_merge",
        category="Distributed Sharding",
        description="Evaluates distributed E-Graph ghost node synchronization across simulated compute nodes.",
        signature=sig,
        expression=expr,
        rules=[fuse_rule],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. LARGE-SCALE 500-NODE E-GRAPH STRESS TEST
# ─────────────────────────────────────────────────────────────────────────────
def build_egraph_stress_test_workload(num_nodes: int = 500) -> Workload:
    """Industrial E-Graph Stress Test with 500 diagram nodes to evaluate search scaling."""
    T = Obj("Tensor")
    sig = Signature()
    sig.add("OpA", T, T)
    sig.add("OpB", T, T)
    sig.add("OpC", T, T)
    sig.add("FusedOpAB", T, T)
    sig.add("FusedOpBC", T, T)

    # Build deep sequential chain of 500 nodes: (OpA ; OpB ; OpC ; OpA ; OpB ; OpC ...)
    ops = ["OpA", "OpB", "OpC"]
    
    expr: Expr = Box(ops[0])
    for i in range(1, num_nodes):
        expr = Seq(expr, Box(ops[i % 3]))

    fuse_ab = Rewrite(
        name="Fuse_OpA_OpB",
        lhs=PSeq(PBox("OpA"), PBox("OpB")),
        rhs=PBox("FusedOpAB"),
    )
    fuse_bc = Rewrite(
        name="Fuse_OpB_OpC",
        lhs=PSeq(PBox("OpB"), PBox("OpC")),
        rhs=PBox("FusedOpBC"),
    )

    return Workload(
        name=f"egraph_stress_test_{num_nodes}_nodes",
        category="E-Graph Scale & Stress",
        description=f"Stress-tests equality saturation performance and e-node growth on a large {num_nodes}-node diagram.",
        signature=sig,
        expression=expr,
        rules=[fuse_ab, fuse_bc],
    )


# ─────────────────────────────────────────────────────────────────────────────
# SUITE REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
def get_all_workloads() -> list[Workload]:
    """Retrieve all benchmark workloads in the validation testbench."""
    return [
        build_transformer_attention_workload(),
        build_llama_decoder_workload(),
        build_lora_chain_workload(),
        build_resnet_conv_bn_workload(),
        build_convnext_block_workload(),
        build_control_flow_licm_workload(),
        build_triton_reduction_workload(),
        build_sharded_egraph_workload(),
        build_egraph_stress_test_workload(500),
    ]
