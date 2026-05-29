"""
    HexSBPSAT

SBP-SAT spectral-element operators on conforming hex meshes, sitting on
top of `HexMeshes` for topology and parametric maps. Equation-agnostic:
the only public PDE building blocks are the discrete Laplacian
`apply_laplacian3d!` and its diagnostics. Wave-equation-specific code
(`Params3d`, initial conditions, Sommerfeld BC, `recommended_dt`) lives
downstream in `WaveToySecondOrder`.

# Layers

* `operators.jl`         — `SBPOps`, reference-element + 1D-domain
                            constructors, `make_operators`, and the
                            `_sat_increment` primitive shared between
                            the 1D and 3D kernels.
* `kernels1d.jl`         — 1D `apply_laplacian!` (per-element + global)
                            and the diagnostic `build_global_laplacian`.
* `geometry.jl`          — `MeshGeometry`, `make_geometry` (for both
                            `HexMesh` and `InflatedCubeMesh`), device
                            migration via `to_device`, and the
                            `Adapt.adapt_structure` rules used by KA at
                            launch time.
* `kernels3d.jl`         — `apply_laplacian3d!`, the two `@kernel`s
                            behind it (face-trace gather + volume work
                            + face SAT + mass division), and the
                            diagnostics built on top of it
                            (`discrete_laplacian`,
                            `spectral_radius_estimate`,
                            `discrete_inner_product`, …).
"""
module HexSBPSAT

using Adapt
using FastGaussQuadrature
using HexMeshes
using HexMeshes: HexMesh, InflatedCubeMesh, MeshConnectivity, PatchInfo,
                 make_cubical_mesh, make_cubed_cube_mesh, make_inflated_cube_mesh,
                 nv, element_vertices, locate_point, invert_element_map,
                 interpolate_field,
                 trilinear_shape, trilinear_dshape,
                 trilinear_map, trilinear_jacobian,
                 lagrange_basis, tensor_interp
# `_patch_point_and_jac` is internal to `HexMeshes` but is needed by
# `make_geometry(::InflatedCubeMesh, elem)` to evaluate the analytic
# Jacobian on the curvilinear patches; `_neigh_pq` is also internal
# but is read by `kernels3d.jl::_face_sat_compute!` to walk the D₄
# orientation transform across an interior face. Pull both in
# explicitly.
using HexMeshes: _patch_point_and_jac, _neigh_pq
using KernelAbstractions
using KrylovKit
using LinearAlgebra
using PolynomialBases: LobattoLegendre
using Random
using StaticArrays

include("operators.jl")
include("kernels1d.jl")
include("geometry.jl")
include("kernels3d.jl")

export
    # Reference element + 1D operators
    make_element, make_domain, make_operators, SBPOps,
    # 1D Laplacian (per-element + global) and diagnostic assembler
    apply_laplacian!, build_global_laplacian,
    # 3D operator-aware geometry
    MeshGeometry, make_geometry, element_coords,
    # Device migration
    to_device,
    # 3D Laplacian + diagnostics
    apply_laplacian3d!,
    discrete_laplacian,
    spectral_radius_estimate,
    discrete_inner_product, discrete_l2_norm,
    physical_mass_diagonal

end # module HexSBPSAT
