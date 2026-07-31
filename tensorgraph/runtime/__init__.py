"""Executable runtime integrations for governed TENSORGRAPH backends."""

from .native_cuda import (
    NativeCUDAExecutable,
    NativeCUDAGraph,
    compile_native_cuda,
)

__all__ = ["NativeCUDAExecutable", "NativeCUDAGraph", "compile_native_cuda"]
