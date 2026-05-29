module HexSBPSATMultiFloatsExt

# Package extension that opts `MultiFloats.MultiFloat` precisions
# (Float32x2, Float64x2, …) out of the native
# `PolynomialBases.LobattoLegendre(N - 1, T)` build path in
# `HexSBPSAT._gll_basis`.
#
# MultiFloats defines `cos(::MultiFloat)` but the method body throws
# an instructive error asking the user to opt into a BigFloat-backed
# transcendental implementation. `LobattoLegendre`'s Newton initial
# guess uses `cos`, so the native build crashes. The fallback in
# `_gll_basis` (build in Float64, convert) sidesteps the issue.
#
# When MultiFloats eventually ships a working `cos`, delete this
# extension and its `[weakdeps]` / `[extensions]` entries in
# `Project.toml`. The test
#
#     MultiFloats: cos(::MultiFloat) is still broken
#
# in `test/test_precision.jl` will fail at that point, signalling the
# fix.

using HexSBPSAT
using MultiFloats: MultiFloat

@inline HexSBPSAT._supports_lobatto_native(::Type{<:MultiFloat}) = false

end # module HexSBPSATMultiFloatsExt
