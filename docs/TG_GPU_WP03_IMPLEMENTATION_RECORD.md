# TG-GPU-WP03 implementation record

Issue: #7

Base implementation identity: `49e589df6863f5549b0c5fe5fc4a5a4bf9d798a4`.

This branch implements the first bounded native-CUDA inference package:

- typed native CUDA source emission for `float16`, `bfloat16`, and `float32`;
- exact-source SHA-256 retention;
- compile and load through PyTorch CUDA extensions;
- ordinary current-stream launch with explicit ABI checks;
- static-buffer CUDA Graph capture and replay;
- changed-input replay differential tests;
- raw evidence schema and fail-closed validator;
- eager, Inductor, generated Triton, and direct Triton benchmark lanes;
- no complete-transformer, production, or performance-portability claim.

The implementation package does not itself constitute CUDA execution evidence.
Positive native-CUDA or latency claims remain gated on exact-merged-commit T4
and Ampere-or-newer replay and independent artifact admission.
