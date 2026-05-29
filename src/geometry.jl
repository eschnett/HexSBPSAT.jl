# Per-node geometric data for the 3D Laplacian kernel, plus device
# migration (`to_device`) and `Adapt.adapt_structure` rules for
# launch-time kernel migration to GPU backends. Composes the
# `HexMeshes`-owned topology + parametric map with the SBP reference-
# element data from `operators.jl` to materialise the per-node
# Jacobian, |det J|, and physical-mass scratch the kernel reads.
#
# Imports `HexMesh`, `InflatedCubeMesh`, `MeshConnectivity`,
# `_patch_point_and_jac`, `element_vertices`, `trilinear_map`,
# `trilinear_jacobian` etc. from `HexMeshes` (loaded in
# `HexSBPSAT.jl`).

"""
    MeshGeometry{T, N}

Per-node geometric data for a `HexMesh`, evaluated at the GLL collocation
points of a 1D reference element with `N` nodes. Holds *only* what the
kernel reads — the underlying `HexMesh` topology (vertices and their
indices into the connectivity) is **not** carried here; keep your own
reference to it for host-side queries (`element_vertices`, plotting,
`locate_point`, etc.).

# Fields

* `Ne :: Int` — element count. Mirrors `mesh.Ne` of the originating
  `HexMesh` and is used as the kernel `ndrange`.
* `conn :: MeshConnectivity{MI, MI8}` — the four connectivity matrices
  copied across from `mesh.conn`. Kernel-resident; backed by `Array`
  on the host and by the appropriate device array on GPU backends.
* `coords :: Array{T, 5}` of shape `(3, N, N, N, Ne)` — physical (x, y, z)
  coordinate of every collocation point.
* `jac    :: Array{T, 6}` of shape `(3, 3, N, N, N, Ne)` — Jacobian
  matrix `J[a, b] = ∂xₐ / ∂ξ_b` of the element map at each node.
* `invjac :: Array{T, 6}` — inverse of `J` at each node; supplies
  `∂ξ / ∂x` to operators that need to pull physical gradients back to the
  reference cube.
* `detjac :: Array{T, 4}` of shape `(N, N, N, Ne)` — absolute value of
  `det J`, the per-node volume factor used by the integration weights.
* `Hphys :: Array{T, 4}` of shape `(N, N, N, Ne)` — the per-node
  physical mass `H_ref[i]·H_ref[j]·H_ref[k]·|det J|`. Precomputed
  here so that GPU-portable reductions (`discrete_inner_product`,
  `discrete_l2_norm`, `spectral_radius_estimate`) can run as a single
  `mapreduce` over device arrays without re-deriving the mass per
  node from the 1D quadrature weights and the Jacobian on each call.
* `face_trace :: Array{T, 5}` of shape `(4, N, N, 6, Ne)` — per-
  element face-trace staging buffer used by the two-pass `rhs3d!`
  implementation. Filled by pass 1 with `(u, ∂x u, ∂y u, ∂z u)` at
  each face quadrature node (physical gradient — the local element's
  `J⁻ᵀ` has already been applied), then read by pass 2 across the
  neighbour relation `mesh.conn.neighbour` to compute the face SAT
  contributions. This is workspace, not geometry: its values are
  overwritten on every `rhs3d!` call. Hard-coded for V=1 fields
  (the wave equation); supporting multi-component PDEs in the future
  will replace the leading `4` with `4·V` or grow a sixth axis.
* `handedness :: Vector{Int8}` of length `Ne` — `±1`, the sign of
  `det J` on element `e`. A non-degenerate hex has uniform-sign Jacobian
  throughout, so a single scalar per element captures the handedness;
  `_add_face_sat!` reads this to pick the outward face normal direction
  without a per-face-node test.

The curvilinear-Laplacian kernel composes these with the 1D quadrature
weights from `ops.H` on the fly: per-node physical mass is
`Hphys = H_ref[i] H_ref[j] H_ref[k] · |det J|` and the weak-form stiffness
kernel is `Wmetric = Hphys · (J⁻¹ J⁻ᵀ)`.
"""
# `MeshGeometry{T, N}` is parametrised on the concrete storage types of
# every kernel-read field so it can be device-resident on any backend.
# All fields are bitstype-adaptable, so KA's launch-time recursive
# `adapt` migrates everything to device types in one shot — there is no
# special handling of any host-only field, because there is none.
struct MeshGeometry{T, N, MC, A5, A6, A4, V1}
    Ne         :: Int
    conn       :: MC
    coords     :: A5
    jac        :: A6
    invjac     :: A6
    detjac     :: A4
    Hphys      :: A4
    face_trace :: A5
    handedness :: V1

    function MeshGeometry{T, N}(Ne::Int, conn::MC,
                                coords::A5, jac::A6, invjac::A6,
                                detjac::A4, Hphys::A4,
                                face_trace::A5,
                                handedness::V1) where {T, N, MC, A5, A6, A4, V1}
        new{T, N, MC, A5, A6, A4, V1}(Ne, conn,
                                       coords, jac, invjac,
                                       detjac, Hphys, face_trace, handedness)
    end
end

"""
    make_geometry(mesh, elem) → MeshGeometry{T, N}

Evaluate the trilinear element map of every hex in `mesh` at the GLL
collocation points of the reference element `elem` (using `elem.xs ∈
[0, 1]` as reference coordinates), and bundle the resulting physical
coordinates, Jacobians, inverse Jacobians, and `|det J|` into a
`MeshGeometry`. The returned geometry copies `mesh.conn` by reference;
the caller retains ownership of the `HexMesh` (with its vertex data)
for host-side queries.
"""
function make_geometry(mesh::HexMesh{T}, elem) where {T}
    N  = elem.N
    ξs = elem.xs
    Ne = mesh.Ne

    # 1D GLL quadrature weights, used to build the per-node physical
    # mass `Hphys`. We pull them from a fresh `SBPOps` rather than
    # depending on the user to pass `ops` in — operator construction
    # is `O(N²)` and runs once at mesh setup, so the cost is invisible.
    ops_ref = make_operators(elem)
    H_1d    = SVector{N, T}(ntuple(i -> ops_ref.H[i, i], Val(N)))

    coords     = Array{T, 5}(undef, 3, N, N, N, Ne)
    jac        = Array{T, 6}(undef, 3, 3, N, N, N, Ne)
    invjac     = Array{T, 6}(undef, 3, 3, N, N, N, Ne)
    detjac     = Array{T, 4}(undef, N, N, N, Ne)
    Hphys      = Array{T, 4}(undef, N, N, N, Ne)
    face_trace = Array{T, 5}(undef, 4, N, N, 6, Ne)   # workspace, see struct doc
    handedness = Vector{Int8}(undef, Ne)
    @inbounds for e in 1:Ne
        verts = element_vertices(mesh, e)
        # Sign of det(J) at element corner (ξ = η = ζ = 0). For a
        # non-degenerate hex the sign is uniform throughout, so any
        # single sample point determines the element's handedness.
        J_corner   = trilinear_jacobian(verts, zero(T), zero(T), zero(T))
        handedness[e] = det(J_corner) ≥ 0 ? Int8(1) : Int8(-1)
        for k in 1:N, j in 1:N, i in 1:N
            ξ, η, ζ = ξs[i], ξs[j], ξs[k]
            p  = trilinear_map(verts, ξ, η, ζ)
            J  = trilinear_jacobian(verts, ξ, η, ζ)
            Ji = inv(J)
            dJ = abs(det(J))
            for a in 1:3
                coords[a, i, j, k, e] = p[a]
                for b in 1:3
                    jac[a, b, i, j, k, e]    = J[a, b]
                    invjac[a, b, i, j, k, e] = Ji[a, b]
                end
            end
            detjac[i, j, k, e] = dJ
            Hphys[i, j, k, e]  = H_1d[i] * H_1d[j] * H_1d[k] * dJ
        end
    end
    return MeshGeometry{T, N}(Ne, mesh.conn,
                              coords, jac, invjac, detjac, Hphys, face_trace, handedness)
end

"""
    make_geometry(mesh::InflatedCubeMesh, elem) → MeshGeometry{T, N}

Evaluate the per-element geometric map of every patch in `mesh` at the
GLL collocation points of the reference element `elem`. Dispatches per
element on `mesh.patch_info[e].kind`:

* `kind == 0` (inner cube): trilinear interpolation of the 8 corners —
  identical to `make_geometry(::HexMesh, elem)`.
* `kind == 1..6` (inflation patch): analytic Jacobian from
  `r(s, η, ζ) = (1 - s)·L + s · R₁ / √(1 + η² + ζ²)` evaluated through
  `_patch_point_and_jac`.
* `kind == 7..12` (shell patch): analytic Jacobian from
  `r(ρ) = (1 - ρ)·R₁ + ρ·R₂`, also through `_patch_point_and_jac`.

The returned `MeshGeometry` is interchangeable with one built from a
plain `HexMesh`; downstream kernels are agnostic to the underlying
mesh's curvature.
"""
function make_geometry(mesh::InflatedCubeMesh{T}, elem) where {T}
    N  = elem.N
    ξs = elem.xs
    Ne = mesh.Ne

    ops_ref = make_operators(elem)
    H_1d    = SVector{N, T}(ntuple(i -> ops_ref.H[i, i], Val(N)))

    coords     = Array{T, 5}(undef, 3, N, N, N, Ne)
    jac        = Array{T, 6}(undef, 3, 3, N, N, N, Ne)
    invjac     = Array{T, 6}(undef, 3, 3, N, N, N, Ne)
    detjac     = Array{T, 4}(undef, N, N, N, Ne)
    Hphys      = Array{T, 4}(undef, N, N, N, Ne)
    face_trace = Array{T, 5}(undef, 4, N, N, 6, Ne)
    handedness = Vector{Int8}(undef, Ne)

    Lv  = mesh.L
    R1v = mesh.R1
    R2v = mesh.R2

    @inbounds for e in 1:Ne
        pi = mesh.patch_info[e]
        if HexMeshes.is_cubical(pi.kind)
            # Trilinear path — inner cube
            verts = element_vertices(mesh.base, e)
            J_c   = trilinear_jacobian(verts, zero(T), zero(T), zero(T))
            handedness[e] = det(J_c) ≥ 0 ? Int8(1) : Int8(-1)
            for k in 1:N, j in 1:N, i in 1:N
                ξ, η, ζ = ξs[i], ξs[j], ξs[k]
                p  = trilinear_map(verts, ξ, η, ζ)
                J  = trilinear_jacobian(verts, ξ, η, ζ)
                Ji = inv(J)
                dJ = abs(det(J))
                for a in 1:3
                    coords[a, i, j, k, e] = p[a]
                    for b in 1:3
                        jac[a, b, i, j, k, e]    = J[a, b]
                        invjac[a, b, i, j, k, e] = Ji[a, b]
                    end
                end
                detjac[i, j, k, e] = dJ
                Hphys[i, j, k, e]  = H_1d[i] * H_1d[j] * H_1d[k] * dJ
            end
        else
            # Analytic curvilinear path — inflation / shell patch
            _, J_c = _patch_point_and_jac(pi, T(0.5), T(0.5), T(0.5),
                                          Lv, R1v, R2v)
            handedness[e] = det(J_c) ≥ 0 ? Int8(1) : Int8(-1)
            for k in 1:N, j in 1:N, i in 1:N
                ξ, η, ζ = ξs[i], ξs[j], ξs[k]
                p, J = _patch_point_and_jac(pi, ξ, η, ζ, Lv, R1v, R2v)
                Ji = inv(J)
                dJ = abs(det(J))
                for a in 1:3
                    coords[a, i, j, k, e] = p[a]
                    for b in 1:3
                        jac[a, b, i, j, k, e]    = J[a, b]
                        invjac[a, b, i, j, k, e] = Ji[a, b]
                    end
                end
                detjac[i, j, k, e] = dJ
                Hphys[i, j, k, e]  = H_1d[i] * H_1d[j] * H_1d[k] * dJ
            end
        end
    end
    return MeshGeometry{T, N}(Ne, mesh.conn,
                              coords, jac, invjac, detjac, Hphys, face_trace, handedness)
end

################################################################################
# Device migration

"""
    to_device(mesh::HexMesh, backend) → HexMesh
    to_device(geom::MeshGeometry, backend) → MeshGeometry

Move every kernel-read array of `mesh` / `geom` onto `backend` (a
`KernelAbstractions.Backend` instance — `CPU()`, `CUDABackend()`,
`MetalBackend()`, `ROCBackend()`). For `mesh`, this migrates the four
connectivity matrices; the host-only `vertex_coords` / `vertex_idx`
are left as plain CPU `Matrix`. For `geom`, it migrates `coords`,
`jac`, `invjac`, `detjac`, `handedness`, and the embedded mesh.

The CPU → CPU case is a no-op-shaped copy: every allocation goes
through `KernelAbstractions.allocate(backend, …)` which on the CPU
backend just calls `Array{T}(undef, …)`. Round-tripping through
`to_device(g, CPU())` is therefore a valid smoke test that exercises
the migration path without requiring a GPU.
"""
function to_device(mesh::HexMesh{T}, backend) where {T}
    nb  = KernelAbstractions.allocate(backend, Int32, size(mesh.neighbour))
    nbf = KernelAbstractions.allocate(backend, Int8, size(mesh.neighbour_face))
    ori = KernelAbstractions.allocate(backend, Int8, size(mesh.orientation))
    bdr = KernelAbstractions.allocate(backend, Int8, size(mesh.bdry))
    copyto!(nb,  mesh.neighbour)
    copyto!(nbf, mesh.neighbour_face)
    copyto!(ori, mesh.orientation)
    copyto!(bdr, mesh.bdry)
    # 3D-specific for now; HexSBPSAT generalization to D=1,2 is deferred.
    # The mesh's D parameter is implicit in `mesh::HexMesh` = `Mesh{3}`.
    new_conn = MeshConnectivity{3}(nb, nbf, ori, bdr)
    return HexMesh{T}(mesh.Ne, new_conn, mesh.vertex_coords, mesh.vertex_idx)
end

function to_device(mesh::InflatedCubeMesh{T}, backend) where {T}
    base_dev = to_device(mesh.base, backend)
    return InflatedCubeMesh(base_dev, mesh.patch_info, mesh.L, mesh.R1, mesh.R2,
                            mesh.M, mesh.Mi, mesh.Ms)
end

function to_device(geom::MeshGeometry{T, N}, backend) where {T, N}
    conn_dev = to_device(geom.conn, backend)
    coords  = KernelAbstractions.allocate(backend, T,    size(geom.coords))
    jac     = KernelAbstractions.allocate(backend, T,    size(geom.jac))
    invjac  = KernelAbstractions.allocate(backend, T,    size(geom.invjac))
    detjac  = KernelAbstractions.allocate(backend, T,    size(geom.detjac))
    Hphys   = KernelAbstractions.allocate(backend, T,    size(geom.Hphys))
    ft      = KernelAbstractions.allocate(backend, T,    size(geom.face_trace))
    hand    = KernelAbstractions.allocate(backend, Int8, size(geom.handedness))
    copyto!(coords, geom.coords)
    copyto!(jac,    geom.jac)
    copyto!(invjac, geom.invjac)
    copyto!(detjac, geom.detjac)
    copyto!(Hphys,  geom.Hphys)
    # face_trace is workspace; no host data to copy. Its values are
    # overwritten on every `rhs3d!` call. We still allocate it on the
    # device so the kernels can write into it directly.
    copyto!(hand,   geom.handedness)
    return MeshGeometry{T, N}(geom.Ne, conn_dev,
                              coords, jac, invjac, detjac, Hphys, ft, hand)
end

# `MeshConnectivity` device migration — used both directly (when a
# caller migrates a HexMesh) and indirectly through `to_device(geom)`.
function to_device(conn::MeshConnectivity, backend)
    nb  = KernelAbstractions.allocate(backend, Int32, size(conn.neighbour))
    nbf = KernelAbstractions.allocate(backend, Int8, size(conn.neighbour_face))
    ori = KernelAbstractions.allocate(backend, Int8, size(conn.orientation))
    bdr = KernelAbstractions.allocate(backend, Int8, size(conn.bdry))
    copyto!(nb,  conn.neighbour)
    copyto!(nbf, conn.neighbour_face)
    copyto!(ori, conn.orientation)
    copyto!(bdr, conn.bdry)
    # 3D-specific for now; HexSBPSAT generalization to D=1,2 is deferred.
    return MeshConnectivity{3}(nb, nbf, ori, bdr)
end

# `Adapt.adapt_structure` rules. When KernelAbstractions launches a
# kernel on a GPU backend, it walks each argument with `Adapt.adapt(to,
# arg)` and replaces host arrays with their device representations
# (`CuArray` → `CuDeviceArray`, `MtlArray` → `MtlDeviceArray`, etc.).
# `MeshConnectivity` and `MeshGeometry` participate in that walk by
# recursively adapting every field. No special rule is needed for
# `HexMesh`: it is host-only and never crosses a kernel boundary.
# Preserve the `D` parameter under adapt — KA `adapt_structure` walks
# every field but the connectivity's spatial-dimension type parameter
# must stay attached to the result.
Adapt.adapt_structure(to, c::MeshConnectivity{D}) where {D} = MeshConnectivity{D}(
    Adapt.adapt(to, c.neighbour),
    Adapt.adapt(to, c.neighbour_face),
    Adapt.adapt(to, c.orientation),
    Adapt.adapt(to, c.bdry))

Adapt.adapt_structure(to, geom::MeshGeometry{T, N}) where {T, N} =
    MeshGeometry{T, N}(
        geom.Ne,
        Adapt.adapt(to, geom.conn),
        Adapt.adapt(to, geom.coords),
        Adapt.adapt(to, geom.jac),
        Adapt.adapt(to, geom.invjac),
        Adapt.adapt(to, geom.detjac),
        Adapt.adapt(to, geom.Hphys),
        Adapt.adapt(to, geom.face_trace),
        Adapt.adapt(to, geom.handedness))

"""
    element_coords(mesh, elem) → Array{T, 5}

Thin wrapper that returns just the physical collocation coordinates from
`make_geometry(mesh, elem)`. Prefer `make_geometry` when you also need the
per-node Jacobian.
"""
element_coords(mesh::HexMesh, elem) = make_geometry(mesh, elem).coords
element_coords(mesh::InflatedCubeMesh, elem) = make_geometry(mesh, elem).coords
