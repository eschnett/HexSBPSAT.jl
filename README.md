# HexSBPSAT.jl

SBP-SAT (Summation-By-Parts / Simultaneous-Approximation-Term) spectral-
element operators on conforming hexahedral meshes, plus the supporting
reference-element and per-node geometry types. Sits on top of
[HexMeshes.jl](https://github.com/eschnetter/HexMeshes.jl) for topology
and parametric maps. Equation-agnostic — applies to anything whose RHS
contains a Laplacian (wave, heat, Schrödinger, …); the wave-equation
driver lives downstream in
[WaveToySecondOrder.jl](https://github.com/eschnetter/WaveToySecondOrder.jl).

## What's here

| Concept | Type / function | Notes |
|---|---|---|
| Reference element | `make_element(T, N)` | GLL nodes on `[0, 1]`, `N` per axis |
| SBP operators | `SBPOps`, `make_operators(elem)` | `B`, `G`, `H`, `Hinv`, `D`, `L` and SAT helpers |
| 1D global Laplacian | `apply_laplacian!(ü, u, bL, bR; dom, ops, τ)` | matrix-shape overload; outer Dirichlet |
| Per-node geometry | `MeshGeometry`, `make_geometry(mesh, elem)` | physical coords, Jacobian, |det J|, H_phys, face-trace workspace |
| Device migration | `to_device(mesh, backend)`, `to_device(geom, backend)` | `KernelAbstractions.Backend` |
| 3D Laplacian | `apply_laplacian3d!(ü, u, bdry_values; geom, ops, τ)` | curvilinear-aware, SIPG, two-launch kernel |
| Diagnostics | `discrete_laplacian`, `spectral_radius_estimate`, `discrete_inner_product`, `discrete_l2_norm`, `physical_mass_diagonal` |  |

## Boundary conditions

`apply_laplacian3d!` treats every outer face with `bdry[f, e] != 0` as a
Dirichlet face with value `bdry_values[bdry[f, e]]`. Other boundary
conditions (Sommerfeld, Robin, free) are layered on top by the caller —
see `WaveToySecondOrder.rhs_wave3d!` for an example of a Sommerfeld
dissipative pass added on top of `apply_laplacian3d!`.

## GPU backends

The kernel-resident arrays in `MeshGeometry` and `MeshConnectivity`
adapt under KernelAbstractions' launch-time `Adapt.adapt`. Package
extensions `HexSBPSATCUDAExt` and `HexSBPSATMetalExt` wire in the
CUDA.jl / Metal.jl backends when those packages are loaded.

## License

MIT.
