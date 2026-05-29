# Loaded automatically by Pkg when both `HexSBPSAT` and `Metal` are
# present in the user's environment (Julia 1.9+ extension mechanism;
# declared via `[weakdeps]` and `[extensions]` in `../Project.toml`).
#
# Nothing actually needs to live here right now — the GPU code path in
# `src/kernels3d.jl` uses only generic primitives (`KernelAbstractions`
# + `Adapt`), so a `MetalBackend()`-resident `geom` and `MtlArray` state
# already flow through it without any Metal-specific dispatch. This
# extension file exists so that:
#
#   1. `Metal` is a weak (opt-in) dep rather than a hard requirement,
#      which means non-GPU users (and CI runners without Apple GPUs) do
#      not have to install Metal to load `HexSBPSAT`.
#
#   2. There is a single, discoverable place to hang any future
#      Metal-specific specialisation (e.g. a `to_device` shortcut that
#      uses `Metal.SharedStorage`, or a kernel launch with a Metal-
#      tuned workgroupsize). Drop those methods in here and they will
#      be loaded if and only if the user also `using Metal`s.
module HexSBPSATMetalExt

using HexSBPSAT
using Metal

end
