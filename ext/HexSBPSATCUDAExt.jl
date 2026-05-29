# Loaded automatically by Pkg when both `HexSBPSAT` and `CUDA` are
# present in the user's environment (Julia 1.9+ extension mechanism;
# declared via `[weakdeps]` and `[extensions]` in `../Project.toml`).
#
# Nothing actually needs to live here right now — the GPU code path in
# `src/kernels3d.jl` uses only generic primitives (`KernelAbstractions`
# + `Adapt`), so a `CUDABackend()`-resident `geom` and `CuArray` state
# already flow through it without any CUDA-specific dispatch. This
# extension file exists so that:
#
#   1. `CUDA` is a weak (opt-in) dep rather than a hard requirement,
#      which means non-GPU users (and CI runners without GPUs) do not
#      have to install CUDA to load `HexSBPSAT`.
#
#   2. There is a single, discoverable place to hang any future
#      CUDA-specific specialisation (e.g. a `to_device` shortcut that
#      uses `CUDA.SharedStorage`, or a kernel launch with a CUDA-tuned
#      workgroupsize). Drop those methods in here and they will be
#      loaded if and only if the user also `using CUDA`s.
module HexSBPSATCUDAExt

using HexSBPSAT
using CUDA

end
