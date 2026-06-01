# Per-node geometric data for the 3D Laplacian kernel, plus device
# migration (`to_device`) and `Adapt.adapt_structure` rules for
# launch-time kernel migration to GPU backends. Composes the
# `HexMeshes`-owned topology + parametric map with the SBP reference-
# element data from `operators.jl` to materialise the per-node
# Jacobian, |det J|, and physical-mass scratch the kernel reads.
#
# Imports `HexMesh`, `MeshConnectivity`, `PatchDesc`,
# `_patch_point_and_jac`, `element_vertices`, `trilinear_map`,
# `trilinear_jacobian` etc. from `HexMeshes` (loaded in
# `HexSBPSAT.jl`).

"""
    MeshGeometry{D, T, N}

Per-node geometric data for a `Mesh{D}`, evaluated at the GLL collocation
points of a 1D reference element with `N` nodes. Holds *only* read-only
geometry — kernel scratch buffers live separately in [`MeshWorkspace`].
The underlying `Mesh` topology (vertices and their indices into the
connectivity) is **not** carried here; keep your own reference to it for
host-side queries (`element_vertices`, plotting, `locate_point`, …).

`D ∈ {2, 3}` is the spatial dimension. `D = 1` is reserved for the 1D
operator path in `kernels1d.jl`, which doesn't need a `MeshGeometry`.

# Fields

* `Ne :: Int` — element count. Mirrors `mesh.Ne` and serves as the
  outer kernel `ndrange` factor.
* `conn :: MeshConnectivity{D}` — the four connectivity matrices copied
  across from `mesh.conn`. Kernel-resident; backed by `Array` on the
  host and by the appropriate device array on GPU backends.

The remaining fields are D-shaped arrays whose ndim depends on `D`:

| field        | dtype | D = 3 shape           | D = 2 shape      |
| ------------ | ----- | --------------------- | ---------------- |
| `coords`     | T     | (3, N, N, N, Ne)      | (2, N, N, Ne)    |
| `jac`        | T     | (3, 3, N, N, N, Ne)   | (2, 2, N, N, Ne) |
| `invjac`     | T     | (3, 3, N, N, N, Ne)   | (2, 2, N, N, Ne) |
| `detjac`     | T     | (N, N, N, Ne)         | (N, N, Ne)       |
| `Hphys`      | T     | (N, N, N, Ne)         | (N, N, Ne)       |
| `handedness` | Int8  | (Ne,)                 | (Ne,)            |

* `coords[a, …, e]` — physical-space coordinate `a ∈ 1..D` of every
  collocation point.
* `jac[a, b, …, e]` — Jacobian `J[a, b] = ∂xₐ / ∂ξ_b` of the element
  map at each node.
* `invjac[a, b, …, e]` — `J⁻¹[a, b] = ∂ξₐ / ∂x_b`, used to pull
  physical gradients back to the reference cube/square.
* `detjac[…, e]` — `|det J|`, the per-node volume factor.
* `Hphys[…, e]` — the per-node physical mass
  `(Πᵢ H_ref[…]) · |det J|`. Precomputed at `make_geometry` so that
  GPU-portable reductions (`discrete_inner_product`,
  `discrete_l2_norm`, `spectral_radius_estimate`) can run as a single
  `mapreduce` over device arrays.
* `handedness[e]` — `±1`, the sign of `det J` on element `e`. A non-
  degenerate hex/quad has uniform-sign Jacobian throughout, so a
  single scalar per element captures the handedness; the face-SAT
  helper reads this to pick the outward normal direction without a
  per-face-node test.

The curvilinear-Laplacian kernel composes these with the 1D quadrature
weights from `ops.H` on the fly: per-node physical mass is
`Hphys = (Πᵢ H_ref[…]) · |det J|` and the weak-form stiffness kernel is
`Wmetric = Hphys · (J⁻¹ J⁻ᵀ)`.
"""
# `MeshGeometry{D, T, N, …}` is parametrised on the concrete storage
# types of every kernel-read field so it can be device-resident on any
# backend. `AF`/`AJ`/`AS`/`VH` are the field-, jacobian-, scalar-, and
# handedness-array types — abbreviated to keep the inferred type
# signature readable in stack traces.
struct MeshGeometry{D, T, N, MC, AF, AJ, AS, VH}
    Ne         :: Int
    conn       :: MC
    coords     :: AF
    jac        :: AJ
    invjac     :: AJ
    detjac     :: AS
    Hphys      :: AS
    handedness :: VH

    function MeshGeometry{D, T, N}(Ne::Int, conn::MC,
                                   coords::AF, jac::AJ, invjac::AJ,
                                   detjac::AS, Hphys::AS,
                                   handedness::VH) where {D, T, N, MC, AF, AJ, AS, VH}
        new{D, T, N, MC, AF, AJ, AS, VH}(Ne, conn,
                                          coords, jac, invjac,
                                          detjac, Hphys, handedness)
    end
end

"""
    MeshWorkspace{D, T, N}

Per-element scratch buffers for the Laplacian kernels — values are
overwritten on every operator call, so the workspace is genuinely
write-only between calls. Pair one `MeshWorkspace` with each
`MeshGeometry` you intend to launch the operator against.

# Fields

* `face_trace :: AF` — staging buffer used by the two-pass Laplacian
  kernels. Filled by pass 1 with `(u, ∇u_phys)` at each face quadrature
  node (the local element's `J⁻ᵀ` has already been applied), then read
  by pass 2 across the neighbour relation `mesh.conn.neighbour` to
  compute the face SAT contributions. Shape depends on `D`:

  | D | shape               | channels                |
  | - | ------------------- | ----------------------- |
  | 3 | (4, N, N, 6, Ne)    | `(u, ∂xu, ∂yu, ∂zu)`    |
  | 2 | (3, N, 4, Ne)       | `(u, ∂xu, ∂yu)`         |

  Downstream BC kernels (e.g. WaveToySecondOrder's Sommerfeld pass)
  read `work.face_trace` directly after `apply_laplacian!` has
  populated it.

Allocate one via [`make_workspace(geom)`](@ref). Multiple workspaces
against the same geometry let independent operator calls run without
sharing scratch — useful for matrix-free Jacobian-vector products and
for keeping a separate workspace per ODE-solver stage.
"""
struct MeshWorkspace{D, T, N, AF}
    face_trace :: AF

    function MeshWorkspace{D, T, N}(face_trace::AF) where {D, T, N, AF}
        new{D, T, N, AF}(face_trace)
    end
end

"""
    make_workspace(geom::MeshGeometry{D, T, N}) → MeshWorkspace{D, T, N}

Allocate a fresh `MeshWorkspace` matching `geom`'s backend, element
count, and polynomial order. The buffer's element type is `T` and its
shape depends on `D` — see [`MeshWorkspace`](@ref).
"""
function make_workspace(geom::MeshGeometry{3, T, N}) where {T, N}
    backend = KernelAbstractions.get_backend(geom.coords)
    ft = KernelAbstractions.allocate(backend, T, 4, N, N, 6, geom.Ne)
    return MeshWorkspace{3, T, N}(ft)
end

function make_workspace(geom::MeshGeometry{2, T, N}) where {T, N}
    backend = KernelAbstractions.get_backend(geom.coords)
    ft = KernelAbstractions.allocate(backend, T, 3, N, 4, geom.Ne)
    return MeshWorkspace{2, T, N}(ft)
end


"""
    make_geometry(mesh::Mesh{3, T}, elem) → MeshGeometry{3, T, N}

Evaluate the trilinear element map of every hex in `mesh` at the GLL
collocation points of the reference element `elem` (using `elem.xs ∈
[0, 1]` as reference coordinates), and bundle the resulting physical
coordinates, Jacobians, inverse Jacobians, and `|det J|` into a
`MeshGeometry`. The returned geometry copies `mesh.conn` by reference;
the caller retains ownership of the `HexMesh` (with its vertex data)
for host-side queries.
"""
function make_geometry(mesh::Mesh{3, T}, elem) where {T}
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
    handedness = Vector{Int8}(undef, Ne)

    @inbounds for e in 1:Ne
        pd  = mesh.patch_desc[mesh.patch_id[e]]
        # Trilinear path for `Cubic` and `Wedge`; analytic
        # (`_patch_point_and_jac`) for `Inflation`, `Shell`, and
        # `WarpedCubic`.
        if pd.kind === Cubic || pd.kind === Wedge
            verts = element_vertices(mesh, e)
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
            # Analytic curvilinear path — Inflation / Shell.
            idx = ntuple(d -> Int(mesh.patch_idx[d, e]), Val(3))
            _, J_c = _patch_point_and_jac(pd, idx, T(0.5), T(0.5), T(0.5))
            handedness[e] = det(J_c) ≥ 0 ? Int8(1) : Int8(-1)
            for k in 1:N, j in 1:N, i in 1:N
                ξ, η, ζ = ξs[i], ξs[j], ξs[k]
                p, J = _patch_point_and_jac(pd, idx, ξ, η, ζ)
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
    return MeshGeometry{3, T, N}(Ne, mesh.conn,
                                 coords, jac, invjac, detjac, Hphys, handedness)
end

################################################################################
# Device migration

"""
    to_device(mesh::Mesh{3}, backend) → Mesh{3}
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
function to_device(mesh::Mesh{3, T}, backend) where {T}
    nb  = KernelAbstractions.allocate(backend, Int32, size(mesh.conn.neighbour))
    nbf = KernelAbstractions.allocate(backend, Int8, size(mesh.conn.neighbour_face))
    ori = KernelAbstractions.allocate(backend, Int8, size(mesh.conn.orientation))
    bdr = KernelAbstractions.allocate(backend, Int8, size(mesh.conn.bdry))
    copyto!(nb,  mesh.conn.neighbour)
    copyto!(nbf, mesh.conn.neighbour_face)
    copyto!(ori, mesh.conn.orientation)
    copyto!(bdr, mesh.conn.bdry)
    # 3D-specific for now; HexSBPSAT generalization to D=1,2 is deferred.
    new_conn = MeshConnectivity{3}(nb, nbf, ori, bdr)
    # Host-only fields (`vertex_coords`, `vertex_idx`, patch metadata)
    # stay as plain CPU `Matrix` / `Vector` since they're never read by
    # kernels — only by host-side queries and `make_geometry`.
    return Mesh{3, T}(mesh.Ne, new_conn, mesh.vertex_coords, mesh.vertex_idx;
                      patch_id              = mesh.patch_id,
                      patch_idx             = mesh.patch_idx,
                      patch_desc            = mesh.patch_desc,
                      patch_element_offset  = mesh.patch_element_offset)
end

function to_device(geom::MeshGeometry{D, T, N}, backend) where {D, T, N}
    conn_dev = to_device(geom.conn, backend)
    coords  = KernelAbstractions.allocate(backend, T,    size(geom.coords))
    jac     = KernelAbstractions.allocate(backend, T,    size(geom.jac))
    invjac  = KernelAbstractions.allocate(backend, T,    size(geom.invjac))
    detjac  = KernelAbstractions.allocate(backend, T,    size(geom.detjac))
    Hphys   = KernelAbstractions.allocate(backend, T,    size(geom.Hphys))
    hand    = KernelAbstractions.allocate(backend, Int8, size(geom.handedness))
    copyto!(coords, geom.coords)
    copyto!(jac,    geom.jac)
    copyto!(invjac, geom.invjac)
    copyto!(detjac, geom.detjac)
    copyto!(Hphys,  geom.Hphys)
    copyto!(hand,   geom.handedness)
    return MeshGeometry{D, T, N}(geom.Ne, conn_dev,
                                 coords, jac, invjac, detjac, Hphys, hand)
end

"""
    to_device(work::MeshWorkspace{D, T, N}, backend) → MeshWorkspace{D, T, N}

Allocate a fresh device-resident workspace matching `work`'s shape.
The contents are scratch — no host-to-device copy is performed.
"""
function to_device(work::MeshWorkspace{D, T, N}, backend) where {D, T, N}
    ft = KernelAbstractions.allocate(backend, T, size(work.face_trace))
    return MeshWorkspace{D, T, N}(ft)
end

# `MeshConnectivity` device migration — used both directly (when a
# caller migrates a Mesh) and indirectly through `to_device(geom)`.
# Preserves the spatial-dimension parameter `D` so 2D and 3D meshes
# round-trip correctly.
function to_device(conn::MeshConnectivity{D}, backend) where {D}
    nb  = KernelAbstractions.allocate(backend, Int32, size(conn.neighbour))
    nbf = KernelAbstractions.allocate(backend, Int8, size(conn.neighbour_face))
    ori = KernelAbstractions.allocate(backend, Int8, size(conn.orientation))
    bdr = KernelAbstractions.allocate(backend, Int8, size(conn.bdry))
    copyto!(nb,  conn.neighbour)
    copyto!(nbf, conn.neighbour_face)
    copyto!(ori, conn.orientation)
    copyto!(bdr, conn.bdry)
    return MeshConnectivity{D}(nb, nbf, ori, bdr)
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

Adapt.adapt_structure(to, geom::MeshGeometry{D, T, N}) where {D, T, N} =
    MeshGeometry{D, T, N}(
        geom.Ne,
        Adapt.adapt(to, geom.conn),
        Adapt.adapt(to, geom.coords),
        Adapt.adapt(to, geom.jac),
        Adapt.adapt(to, geom.invjac),
        Adapt.adapt(to, geom.detjac),
        Adapt.adapt(to, geom.Hphys),
        Adapt.adapt(to, geom.handedness))

Adapt.adapt_structure(to, work::MeshWorkspace{D, T, N}) where {D, T, N} =
    MeshWorkspace{D, T, N}(Adapt.adapt(to, work.face_trace))

"""
    make_geometry(mesh::Mesh{2, T}, elem) → MeshGeometry

2D analog of [`make_geometry(::Mesh{3}, elem)`](@ref). Returns a
`MeshGeometry` whose arrays are 2D-shaped:

* `coords     :: Array{T, 4}` of shape `(2, N, N, Ne)`
* `jac        :: Array{T, 5}` of shape `(2, 2, N, N, Ne)`
* `invjac     :: Array{T, 5}` of shape `(2, 2, N, N, Ne)`
* `detjac     :: Array{T, 3}` of shape `(N, N, Ne)`
* `Hphys      :: Array{T, 3}` of shape `(N, N, Ne)`
* `handedness :: Vector{Int8}` of length `Ne`.

Bilinear (`Cubic` / `Wedge` patches) and analytic
(`Inflation` / `Shell`) paths exactly mirror the 3D structure. The
scratch face-trace buffer lives in a separate [`MeshWorkspace`].
"""
function make_geometry(mesh::Mesh{2, T}, elem) where {T}
    N  = elem.N
    ξs = elem.xs
    Ne = mesh.Ne

    ops_ref = make_operators(elem)
    H_1d    = SVector{N, T}(ntuple(i -> ops_ref.H[i, i], Val(N)))

    coords     = Array{T, 4}(undef, 2, N, N, Ne)
    jac        = Array{T, 5}(undef, 2, 2, N, N, Ne)
    invjac     = Array{T, 5}(undef, 2, 2, N, N, Ne)
    detjac     = Array{T, 3}(undef, N, N, Ne)
    Hphys      = Array{T, 3}(undef, N, N, Ne)
    handedness = Vector{Int8}(undef, Ne)

    @inbounds for e in 1:Ne
        pd = mesh.patch_desc[mesh.patch_id[e]]
        if pd.kind === Cubic || pd.kind === Wedge
            verts = element_vertices(mesh, e)
            J_c   = bilinear_jacobian(verts, zero(T), zero(T))
            handedness[e] = det(J_c) ≥ 0 ? Int8(1) : Int8(-1)
            for j in 1:N, i in 1:N
                ξ, η = ξs[i], ξs[j]
                p  = bilinear_map(verts, ξ, η)
                J  = bilinear_jacobian(verts, ξ, η)
                Ji = inv(J)
                dJ = abs(det(J))
                for a in 1:2
                    coords[a, i, j, e] = p[a]
                    for b in 1:2
                        jac[a, b, i, j, e]    = J[a, b]
                        invjac[a, b, i, j, e] = Ji[a, b]
                    end
                end
                detjac[i, j, e] = dJ
                Hphys[i, j, e]  = H_1d[i] * H_1d[j] * dJ
            end
        else
            idx = ntuple(d -> Int(mesh.patch_idx[d, e]), Val(2))
            _, J_c = _patch_point_and_jac_2d(pd, idx, T(0.5), T(0.5))
            handedness[e] = det(J_c) ≥ 0 ? Int8(1) : Int8(-1)
            for j in 1:N, i in 1:N
                ξ, η = ξs[i], ξs[j]
                p, J = _patch_point_and_jac_2d(pd, idx, ξ, η)
                Ji = inv(J)
                dJ = abs(det(J))
                for a in 1:2
                    coords[a, i, j, e] = p[a]
                    for b in 1:2
                        jac[a, b, i, j, e]    = J[a, b]
                        invjac[a, b, i, j, e] = Ji[a, b]
                    end
                end
                detjac[i, j, e] = dJ
                Hphys[i, j, e]  = H_1d[i] * H_1d[j] * dJ
            end
        end
    end
    return MeshGeometry{2, T, N}(Ne, mesh.conn,
                                 coords, jac, invjac, detjac, Hphys, handedness)
end

"""
    to_device(mesh::Mesh{2, T}, backend) → Mesh{2, T}

2D analog of [`to_device(::Mesh{3}, backend)`](@ref).
"""
function to_device(mesh::Mesh{2, T}, backend) where {T}
    nb  = KernelAbstractions.allocate(backend, Int32, size(mesh.conn.neighbour))
    nbf = KernelAbstractions.allocate(backend, Int8, size(mesh.conn.neighbour_face))
    ori = KernelAbstractions.allocate(backend, Int8, size(mesh.conn.orientation))
    bdr = KernelAbstractions.allocate(backend, Int8, size(mesh.conn.bdry))
    copyto!(nb,  mesh.conn.neighbour)
    copyto!(nbf, mesh.conn.neighbour_face)
    copyto!(ori, mesh.conn.orientation)
    copyto!(bdr, mesh.conn.bdry)
    new_conn = MeshConnectivity{2}(nb, nbf, ori, bdr)
    return Mesh{2, T}(mesh.Ne, new_conn, mesh.vertex_coords, mesh.vertex_idx;
                      patch_id              = mesh.patch_id,
                      patch_idx             = mesh.patch_idx,
                      patch_desc            = mesh.patch_desc,
                      patch_element_offset  = mesh.patch_element_offset)
end

"""
    element_coords(mesh, elem) → Array{T, …}

Thin wrapper that returns just the physical collocation coordinates from
`make_geometry(mesh, elem)`. For `Mesh{3}` returns a `(3, N, N, N, Ne)`
array; for `Mesh{2}` a `(2, N, N, Ne)` array.
"""
element_coords(mesh::Mesh{3}, elem) = make_geometry(mesh, elem).coords
element_coords(mesh::Mesh{2}, elem) = make_geometry(mesh, elem).coords
