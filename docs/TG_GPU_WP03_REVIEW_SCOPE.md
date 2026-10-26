# TG-GPU-WP03 review scope

Review only the bounded native-CUDA unary inference package. Confirm that the
exact generated kernel source is compiled and launched without substitution,
that ABI and graph-capture failures fail closed, that changed input contents are
verified after capture, and that timing categories remain separated.

Do not promote complete transformer decoding, production readiness, or portable
performance from CPU CI or source inspection.
