# TG-GPU-WP03 implementation checklist

- [x] Historical CUDA-emitter lineage retained.
- [x] Exact generated source and SHA-256 retained.
- [x] `float16`, `bfloat16`, and `float32` storage specializations emitted.
- [x] Unsupported graph forms fail closed.
- [x] Domain-controlled `Log` fails closed without a positivity proof.
- [x] PyTorch CUDA-extension compile and load path provided.
- [x] Current-stream ordinary launch provided.
- [x] Shape, dtype, device, layout, and alias checks provided.
- [x] Stable-buffer CUDA Graph capture and replay provided.
- [x] Changed-input graph replay test provided.
- [x] Raw CUDA-event evidence runner provided.
- [x] Evidence schema and standard-library validator provided.
- [x] Adversarial validator mutation tests provided.
- [ ] Exact merged implementation replayed on T4.
- [ ] Exact merged implementation replayed on Ampere-or-newer.
- [ ] Independent direct-native baseline admitted.
- [ ] Raw artifacts independently reviewed and admitted.
