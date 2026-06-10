"""
    HexSBPSAT

SBP-SAT spectral-element operators on conforming hex meshes, sitting on
top of `HexMeshes` for topology and parametric maps. Equation-agnostic:
the only public PDE building blocks are the discrete Laplacian
`apply_laplacian!`, the 1D first-derivative operator `apply_D!`, and
their diagnostics. Wave-equation-specific code
(`Params3d`, initial conditions, Sommerfeld BC, `recommended_dt`) lives
downstream in `WaveToySecondOrder`.

# Layers

* `operators.jl`         — `SBPOps`, reference-element + 1D-domain
                            constructors, `make_operators`, and the
                            `_sat_increment` primitive shared between
                            the 1D and 3D kernels.
* `kernels1d.jl`         — 1D `apply_laplacian!` (per-element + global),
                            the connectivity-driven first-derivative
                            operator `apply_D!`, and the diagnostic
                            `build_global_laplacian`.
* `geometry.jl`          — `MeshGeometry`, `make_geometry` (dispatches
                            on per-element `PatchDesc.kind` to choose
                            trilinear vs analytic-Jacobian path),
                            device migration via `to_device`, and the
                            `Adapt.adapt_structure` rules used by KA at
                            launch time.
* `kernels3d.jl`         — `apply_laplacian!`, the two `@kernel`s
                            behind it (face-trace gather + volume work
                            + face SAT + mass division), and the
                            diagnostics built on top of it
                            (`discrete_laplacian`,
                            `spectral_radius_estimate`,
                            `discrete_inner_product`, …).
"""
module HexSBPSAT

using Adapt
using HexMeshes
using HexMeshes: Mesh, MeshConnectivity, PatchDesc, PatchKind,
                 Cubic, Wedge, Inflation, Shell, WarpedCubic,
                 make_uniform_quad, make_cubed_square_mesh, make_inflated_square_mesh,
                 make_uniform_hex, make_cubed_cube_mesh, make_inflated_cube_mesh,
                 nv, npatches, element_vertices, locate_point, invert_element_map,
                 interpolate_field,
                 linear_map, linear_jacobian,
                 bilinear_shape, bilinear_dshape,
                 bilinear_map, bilinear_jacobian,
                 trilinear_shape, trilinear_dshape,
                 trilinear_map, trilinear_jacobian,
                 lagrange_basis, tensor_interp
# `_patch_point_and_jac` is internal to `HexMeshes` but is needed by
# `make_geometry` to evaluate the analytic Jacobian on the curvilinear
# patches; `_neigh_pq` / `_neigh_p` are also internal but are read by
# `kernels3d.jl::_face_sat_compute!` / `kernels2d.jl::_face_sat_compute_2d!`
# to walk the orientation transform across an interior face. Pull them
# in explicitly.
using HexMeshes: _patch_point_and_jac, _patch_point_and_jac_2d,
                 _neigh_p, _neigh_pq
using KernelAbstractions
using KrylovKit
using LinearAlgebra
using PolynomialBases: LobattoLegendre
using Random
using StaticArrays

# `geometry.jl` precedes `kernels1d.jl` because the connectivity-driven
# 1D operator `apply_D!` dispatches on `MeshGeometry{1}`.
include("operators.jl")
include("geometry.jl")
include("kernels1d.jl")
include("kernels3d.jl")
include("kernels2d.jl")
# Multi-dimensional axis-selective first derivative (`apply_D!(…, d)`),
# the building block for the conservative first-order scalar wave.
include("kernels2d_grad.jl")
# 3D axis-selective first derivative `apply_D!(…, d)` (affine).
include("kernels3d_grad.jl")
# Curvilinear 2D conservative gradient/divergence (free-stream-
# preserving, discrete metric terms).
include("kernels2d_curv.jl")
# Curvilinear 3D conservative metric terms (conservative-curl form) +
# gradient/divergence.
include("kernels3d_curv.jl")
# Channel-batched 3D operators (one launch for C fields; multi-field
# systems like the generalized harmonic equations).
include("kernels3d_batch.jl")

export
    # Reference element + 1D operators
    make_element, make_domain, make_operators, SBPOps,
    # Per-element / 1D-global Laplacian + diagnostic assembler
    build_global_laplacian,
    # Connectivity-driven 1D first derivative (SBP-G + centred-flux SAT)
    apply_D!,
    # Curvilinear 2D conservative gradient/divergence + metric terms
    make_metric_terms2d, make_metric_terms3d, metric_to_device,
    apply_gradient2d!, apply_divergence2d!,
    apply_gradient3d!, apply_divergence3d!,
    # Channel-batched 3D operators (trailing channel dimension)
    apply_D_batch!, apply_gradient3d_batch!, apply_divergence3d_batch!,
    # Operator-aware geometry (dimension-generic in `D ∈ {2, 3}`) +
    # the per-call scratch workspace that goes with it.
    MeshGeometry, make_geometry, element_coords,
    MeshWorkspace, make_workspace,
    # Device migration
    to_device,
    # Dimension-generic Laplacian + diagnostics. `apply_laplacian!`
    # dispatches on `MeshGeometry{D, T, N}` (D = 2 or 3) for the
    # curvilinear path, and on `AbstractVector` / `AbstractMatrix` for
    # the 1D per-element / 1D-global paths in `kernels1d.jl`.
    apply_laplacian!,
    discrete_laplacian,
    spectral_radius_estimate, eigsolve_default_tol,
    discrete_inner_product, discrete_l2_norm,
    physical_mass_diagonal

end # module HexSBPSAT
