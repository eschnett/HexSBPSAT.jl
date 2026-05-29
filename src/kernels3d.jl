# Curvilinear-aware 3D discrete Laplacian on a `MeshGeometry` (built on
# top of a `HexMesh` or `InflatedCubeMesh`). Single public entry point
# `apply_laplacian3d!(ü, u, bdry_values; geom, ops, τ)` — a pure
# spatial operator with outer Dirichlet boundary conditions and an
# SIPG interior face SAT. Equation-free: this file knows nothing
# about wave equations, time evolution, or `u̇`. Other boundary
# conditions (Sommerfeld, Robin, Neumann) are layered on by downstream
# packages via post-passes that read the SAT face geometry from `geom`.
#
# Element shape enters only through the per-node Jacobian carried by the
# `MeshGeometry`. The kernel applies the same machinery to axis-aligned
# cubes (where the Jacobian is a constant diagonal) and to curved hex
# elements (where the Jacobian varies per node and may be dense).
#
# Per-element structure
# ---------------------
# For each element `e`, the kernel computes `(L u)|_e` where `L` is the
# discrete physical Laplacian:
#
#   1. **Reference-axis gradients.** Compute `∂_a u[i,j,k]` for `a = 1,2,3`
#      via three 1D `G`-sweeps along each reference axis.
#
#   2. **Weak stiffness flux.** At each node, form
#         W_a[i,j,k] = H_ref[i] H_ref[j] H_ref[k] · |det J| ·
#                        Σ_b (J⁻¹ J⁻ᵀ)_{ab} · ∂_b u.
#      The contravariant metric `J⁻¹ J⁻ᵀ` couples reference axes
#      whenever `J` is non-diagonal (curvilinear elements).
#
#   3. **Weak volume term.** Apply `-Gᵀ` along each reference axis to
#      `W_a` and sum into `(H_phys · L u)`. This is the volume part of
#      the SBP-DG weak Laplacian; for axis-aligned cubes it reduces to
#      `(1/h²) · (−H⁻¹ Gᵀ H G) u` along each axis.
#
#   4. **Per-face SBP-SAT terms** (six faces per element). At every face
#      quadrature node, the kernel:
#         (a) builds the two tangent vectors as columns of `J` and takes
#             their cross product → physical outward normal `n_phys` and
#             surface element `|J_F|`;
#         (b) forms `μ = J⁻¹ · n_phys` (per-face-node 3-vector);
#         (c) evaluates the physical normal gradient
#                ∇u · n_phys = Σ_c μ_c · ∂_c u
#             on both the self and (when interior) neighbour sides;
#         (d) adds, with face quadrature weight `w_F = |J_F| · H_face`,
#                + w_F · (∇u · n_phys)_self            ← boundary lift
#                − (1/2) · w_F · Δ(∇u · n_phys)        ← adjoint consistency
#                − τ · w_F · Δu                        ← penalty
#             to the face node, and the symmetric SIPG lift
#                + α · w_F · Δu  ·  Σ_c μ_c · Gᵀ_c
#             distributed via three reference-axis `Gᵀ` lifts. For
#             axis-aligned cubes only `μ_a` (the orthogonal axis) is
#             non-zero, so this collapses to the existing single-axis
#             lift used in earlier code.
#
#   5. **Mass division.** Divide the accumulated `H_phys · L u` by the
#      per-node physical mass `H_phys = H_ref · |det J|` to recover
#      `L u` in physical units.

using KernelAbstractions: @kernel, @index, @Const,
                          get_backend, synchronize

################################################################################
# Face axis bookkeeping
#
# Face index convention (matches `HexMesh`):
#   1 = −x    2 = +x    3 = −y    4 = +y    5 = −z    6 = +z
#
# For face `f`, the orthogonal reference axis is `a = (f+1) ÷ 2`. The two
# in-face tangent reference axes `(axis_p, axis_q)` are picked so that the
# face quadrature node index `(p, q)` maps naturally to the volume nodes:
#
#   a = 1 → (axis_p, axis_q) = (2, 3),  volume node = (face_row, p, q)
#   a = 2 → (axis_p, axis_q) = (1, 3),  volume node = (p, face_row, q)
#   a = 3 → (axis_p, axis_q) = (1, 2),  volume node = (p, q, face_row)
#
# `face_row = 1` on a −face, `N` on a +face. The natural cross product
# `t_axis_p × t_axis_q` equals `+ê_a` for `a ∈ {1,3}` and `−ê_a` for
# `a = 2`; the kernel multiplies by `cross_sign · face_sign` so the
# resulting vector points outward from the element.

# Per-kernel small-integer indexes are `Int32` throughout. NVIDIA GPUs
# emulate `Int64` arithmetic as pairs of 32-bit ops (≈4× slower per
# multiply on register-bound code), and Apple Silicon GPUs are
# similarly 32-bit-native. The element index `e` is the one
# intentional exception — meshes might (eventually) exceed 2³¹
# elements. Everything else (workitem indices, loop bounds, face
# axis bookkeeping) stays in Int32. The `Int32(...)` casts here are
# compile-time constants and get folded by the compiler.
#
# `_face_axis_idx` and `_face_row` return `Int` (not `Int32`) because
# their results are immediately used as `Val{…}` type parameters and
# `Val{1}::Type` doesn't match `Val{Int32(1)}::Type`. The values
# themselves are tiny (1–6, 1 or N), so the Int64-arithmetic cost on
# downstream comparisons is negligible.
@inline _face_axis_idx(::Val{f}) where {f} = (f + 1) ÷ 2
@inline _face_row(::Val{f}, ::Val{N}) where {f, N} = isodd(f) ? 1 : N
@inline _face_sign(::Val{f}, ::Type{T}) where {f, T} = isodd(f) ? -one(T) : one(T)
@inline _cross_sign(::Val{a}, ::Type{T}) where {a, T} = a == 2 ? -one(T) : one(T)
@inline _tangent_axes(::Val{1}) = (Int32(2), Int32(3))
@inline _tangent_axes(::Val{2}) = (Int32(1), Int32(3))
@inline _tangent_axes(::Val{3}) = (Int32(1), Int32(2))

# Volume index of face node `(p, q)` for face axis `a` and `face_row`.
# Inputs are typed Int32, outputs are Int32 — keep the index-arithmetic
# chain narrow.
@inline _face_volume_idx(::Val{1}, face_row, p, q) = (face_row, p, q)
@inline _face_volume_idx(::Val{2}, face_row, p, q) = (p, face_row, q)
@inline _face_volume_idx(::Val{3}, face_row, p, q) = (p, q, face_row)

# Runtime variant used for the neighbour lookup (the neighbour's face
# axis is known only at runtime from `mesh.conn.neighbour_face`).
@inline function _face_volume_idx(a::Integer, face_row, p, q)
    if a == 1
        return (face_row, p, q)
    elseif a == 2
        return (p, face_row, q)
    else
        return (p, q, face_row)
    end
end

# `_neigh_pq` is defined in `mesh.jl`.

# Decode workitem-local `(i, j, k) ∈ Int32(1):Int32(N)` from KA's
# `@index(Local, Linear)`. Backends return `Int` (CPU) or `Int32`
# (Metal/CUDA) — we narrow to `Int32` via `li % Int32` (a non-overflowing
# truncating cast for any `li ≤ 2³¹`, which is always true since the
# workgroup is at most `N³` workitems with `N ≤ 17` or so).
#
# Called fresh at the top of every post-`@synchronize` phase, because
# locals don't survive a barrier on the CPU backend. The compiler
# inlines this away.
@inline function _ijk_from_li(li, ::Val{N}) where {N}
    li0 = (li % Int32) - Int32(1)
    n   = Int32(N)
    i   = (li0 % n) + Int32(1)
    j   = ((li0 ÷ n) % n) + Int32(1)
    k   = (li0 ÷ (n * n)) + Int32(1)
    return i, j, k
end

################################################################################
# Face-trace gather (pass 1) + per-face SAT (pass 2)
#
# The kernel is split into two launches that share an element-indexed
# global staging buffer `geom.face_trace`. This converts the face-SAT's
# scattered neighbour reads (`u_global[..., nbr]` repeated for every
# face quadrature node and every stencil component) into a single
# coalesced load of `4·N²` floats per face per element. The two
# passes are:
#
#   Pass 1 (volume + trace): compute reference gradients `Du`,
#     weak stiffness flux `W`, and write the volume divergence to
#     `ü` (before mass division). Additionally, for each of the
#     element's six faces, gather the trace data — the scalar field
#     `u` and the **physical** gradient `∇u = J⁻ᵀ · Du` — into
#     `face_trace[1:4, p, q, f, e]`. We store the physical gradient
#     rather than the reference one so that pass 2 doesn't need to
#     read the neighbour's `geom.invjac` to reconstruct it.
#
#   Pass 2 (face SAT + mass div): add the four face SAT terms to
#     `ü`, reading self's data from `face_trace[..., self_face, e]`
#     and the neighbour's from `face_trace[..., nbr_face, nbr]`.
#     The only indirected read is the neighbour index per face — six
#     pointer fetches per element — and the actual face data is
#     contiguous in `face_trace`. Finally divide by `Hphys` per node.
#
# KA's command queue is FIFO per backend, so launching pass 2 after
# pass 1 guarantees pass 2 sees the trace data pass 1 wrote, without
# an explicit `synchronize`.

# Pass 2 face-SAT helpers.
#
# Pass 2 is workgroup-per-element: each workgroup has N³ workitems
# (one per collocation node) and shares a small `face_buf` in
# `@localmem` for one face at a time. For each of the six faces we
# run two phases separated by `@synchronize`:
#
#   1. *Compute* (`_face_sat_compute!`): only the N² workitems
#      whose `(i, j, k)` lies on the face are active. Each one
#      computes its face-node's SAT contributions and packs them into
#      `face_buf[1..4, p_local, q_local]`:
#         slot 1 — `bcorr = wF · (Gn_self − ½ ΔGn − τ Δu/h_F)`,
#                  the boundary lift + adjoint + penalty correction;
#         slot 2 — `μ_a · ν`, the weight for the "interior" SIPG lift
#                  along the face's orthogonal axis `a`;
#         slot 3 — `μ_axis_p · ν`, for the in-face lift along axis_p;
#         slot 4 — `μ_axis_q · ν`, for the in-face lift along axis_q.
#      Off-face workitems return without writing anything.
#
#   2. *Apply* (`_face_sat_apply!`): every workitem reads its
#      contribution from `face_buf` and updates its own
#      `üe_loc[i, j, k]`. The "interior" lift hits every workitem;
#      the two "in-face" lifts and the `bcorr` only hit workitems
#      that are themselves on the face (`ia == face_r`).
#
# The gather formulation eliminates all the scatter writes the old
# one-workitem-per-element pass 2 used to do — the SIPG lift's
# `üe[l, j, k] += G[i, l]·μ·ν` was a write race waiting to happen on
# any parallelization across face nodes. Here each workitem only
# writes `üe_loc[i, j, k]` (its own slot), so no atomics are needed.

@inline function _face_sat_compute!(::Val{f}, face_buf,
                                       i, j, k, e,
                                       geom::MeshGeometry{T, N},
                                       conn::MeshConnectivity,
                                       ops::SBPOps{N, T}, τ::T,
                                       bdry_values::NTuple{6, T},
                                       H_1d::SVector{N, T},
                                       ::Val{N}) where {f, T, N}
    a_idx          = _face_axis_idx(Val(f))
    face_r         = _face_row(Val(f), Val(N))
    sgn_f          = _face_sign(Val(f), T)
    sgn_c          = _cross_sign(Val(a_idx), T)
    axis_p, axis_q = _tangent_axes(Val(a_idx))

    # Is the workitem on this face? (i.e., its axis-a component == face_r)
    ia = a_idx == 1 ? i : a_idx == 2 ? j : k
    ia == face_r || return nothing

    # Face-local 2D coordinates extracted from the workitem's volume
    # coordinates. By construction the workitem's `(i, j, k)` IS the
    # face node's volume position.
    p_local, q_local = a_idx == 1 ? (j, k) :
                       a_idx == 2 ? (i, k) :
                                    (i, j)

    # Self J at this workitem's position (= the face node).
    @inbounds J11 = geom.jac[1,1,i,j,k,e]; @inbounds J12 = geom.jac[1,2,i,j,k,e]; @inbounds J13 = geom.jac[1,3,i,j,k,e]
    @inbounds J21 = geom.jac[2,1,i,j,k,e]; @inbounds J22 = geom.jac[2,2,i,j,k,e]; @inbounds J23 = geom.jac[2,3,i,j,k,e]
    @inbounds J31 = geom.jac[3,1,i,j,k,e]; @inbounds J32 = geom.jac[3,2,i,j,k,e]; @inbounds J33 = geom.jac[3,3,i,j,k,e]

    tp_x = axis_p == 1 ? J11 : axis_p == 2 ? J12 : J13
    tp_y = axis_p == 1 ? J21 : axis_p == 2 ? J22 : J23
    tp_z = axis_p == 1 ? J31 : axis_p == 2 ? J32 : J33
    tq_x = axis_q == 1 ? J11 : axis_q == 2 ? J12 : J13
    tq_y = axis_q == 1 ? J21 : axis_q == 2 ? J22 : J23
    tq_z = axis_q == 1 ? J31 : axis_q == 2 ? J32 : J33

    @inbounds sgn_out = sgn_f * sgn_c * T(geom.handedness[e])

    nx_u = sgn_out * (tp_y * tq_z - tp_z * tq_y)
    ny_u = sgn_out * (tp_z * tq_x - tp_x * tq_z)
    nz_u = sgn_out * (tp_x * tq_y - tp_y * tq_x)
    JF = sqrt(nx_u*nx_u + ny_u*ny_u + nz_u*nz_u)
    nx = nx_u / JF; ny = ny_u / JF; nz = nz_u / JF
    # SIPG penalty length scale: the perpendicular element thickness
    # `h_F = |K_e| / |F|`, i.e. `detjac / JF` evaluated at this face
    # node. (NOT `sqrt(JF)`, which is the face geometric mean — for
    # axis-aligned uniform elements the two coincide, but on stretched
    # elements `sqrt(JF)` under-penalises by exactly the aspect ratio.
    # This was the source of the +30 000 localised eigenvalue on the
    # outermost layer of the inflated-cube shell at M = 8.)
    @inbounds dJ = geom.detjac[i, j, k, e]
    hF = dJ / JF

    @inbounds Ji11 = geom.invjac[1,1,i,j,k,e]; @inbounds Ji12 = geom.invjac[1,2,i,j,k,e]; @inbounds Ji13 = geom.invjac[1,3,i,j,k,e]
    @inbounds Ji21 = geom.invjac[2,1,i,j,k,e]; @inbounds Ji22 = geom.invjac[2,2,i,j,k,e]; @inbounds Ji23 = geom.invjac[2,3,i,j,k,e]
    @inbounds Ji31 = geom.invjac[3,1,i,j,k,e]; @inbounds Ji32 = geom.invjac[3,2,i,j,k,e]; @inbounds Ji33 = geom.invjac[3,3,i,j,k,e]
    μ1 = Ji11*nx + Ji12*ny + Ji13*nz
    μ2 = Ji21*nx + Ji22*ny + Ji23*nz
    μ3 = Ji31*nx + Ji32*ny + Ji33*nz

    @inbounds wF = JF * H_1d[p_local] * H_1d[q_local]

    @inbounds u_self  = geom.face_trace[1, p_local, q_local, f, e]
    @inbounds gx_self = geom.face_trace[2, p_local, q_local, f, e]
    @inbounds gy_self = geom.face_trace[3, p_local, q_local, f, e]
    @inbounds gz_self = geom.face_trace[4, p_local, q_local, f, e]
    Gn_self = nx * gx_self + ny * gy_self + nz * gz_self

    @inbounds nbr = conn.neighbour[f, e]
    @inbounds tag = conn.bdry[f, e]
    α = nbr == 0 ? one(T) : one(T) / 2

    Δu::T  = zero(T)
    ΔGn::T = zero(T)
    if nbr == 0
        # Outer face. Tags 1..6 → Dirichlet drive against
        # `bdry_values[tag]`. Every other tag (including 0 and the
        # downstream Sommerfeld tag) is a "free" face: this operator
        # contributes only the *natural* boundary lift `wF · Gn_self`
        # (the discrete analog of the `∮ψ ∂_n u dS` IBP term), with no
        # Dirichlet drive and no SIPG penalty. The downstream caller
        # layers its own BC (Sommerfeld, Robin, Neumann, …) on top by
        # post-passing over those faces and reading the geometry +
        # face traces directly. `Δu = ΔGn = 0` here makes `bcorr`
        # below collapse to exactly `wF · Gn_self` and zeros out the
        # SIPG lifts via `ν = α · wF · Δu = 0`.
        if Int8(1) ≤ tag ≤ Int8(6)
            @inbounds u_neigh = bdry_values[tag]
            Δu  = u_self - u_neigh
        end
    else
        @inbounds nbr_face = Int32(conn.neighbour_face[f, e])
        @inbounds nbr_o    = conn.orientation[f, e]
        pn, qn = _neigh_pq(nbr_o, p_local, q_local, Int32(N))
        @inbounds u_neigh = geom.face_trace[1, pn, qn, nbr_face, nbr]
        @inbounds gx_n    = geom.face_trace[2, pn, qn, nbr_face, nbr]
        @inbounds gy_n    = geom.face_trace[3, pn, qn, nbr_face, nbr]
        @inbounds gz_n    = geom.face_trace[4, pn, qn, nbr_face, nbr]
        Gn_neigh = nx * gx_n + ny * gy_n + nz * gz_n
        Δu  = u_self  - u_neigh
        ΔGn = Gn_self - Gn_neigh
    end

    ν   = α * wF * Δu

    # Pick the right μ component for each lift axis.
    μa = a_idx  == 1 ? μ1 : a_idx  == 2 ? μ2 : μ3
    μp = axis_p == 1 ? μ1 : axis_p == 2 ? μ2 : μ3
    μq = axis_q == 1 ? μ1 : axis_q == 2 ? μ2 : μ3

    bcorr = wF * (Gn_self - (T(1)/2) * ΔGn - τ * Δu / hF)

    @inbounds face_buf[1, p_local, q_local] = bcorr
    @inbounds face_buf[2, p_local, q_local] = μa * ν
    @inbounds face_buf[3, p_local, q_local] = μp * ν
    @inbounds face_buf[4, p_local, q_local] = μq * ν
    return nothing
end

@inline function _face_sat_apply!(::Val{f}, üe_loc, face_buf,
                                     i, j, k,
                                     ops::SBPOps{N, T},
                                     ::Val{N}) where {f, T, N}
    a_idx  = _face_axis_idx(Val(f))
    face_r = _face_row(Val(f), Val(N))

    # The destination's projection onto the face axes. `ia` is the
    # workitem's axis-a coordinate (matched against `face_r` for the
    # "in-face" lifts and bcorr). `p_local`, `q_local` extract the
    # workitem's face-plane coordinates (which feed the interior-
    # lift's gather of `face_buf[2, p_local, q_local]`).
    ia = a_idx == 1 ? i : a_idx == 2 ? j : k
    p_local, q_local = a_idx == 1 ? (j, k) :
                       a_idx == 2 ? (i, k) :
                                    (i, j)

    # Interior lift along axis `a`: spreads `μ_a · ν` of each face
    # node out to a 1-D line through the volume in the axis-a
    # direction. Every workitem receives a contribution; the weight
    # `G[face_r, ia]` selects the workitem's position on that line.
    @inbounds üe_loc[i, j, k] += ops.G[face_r, ia] * face_buf[2, p_local, q_local]

    # In-face lifts + boundary correction — only fire for workitems
    # actually on the face. The two gather sums replace the original
    # scatter writes `üe[i, l, k] += G[j, l]·μ·ν` (which would race
    # across face workitems on a workgroup-per-element layout).
    if ia == face_r
        s_p = zero(T)
        s_q = zero(T)
        @inbounds for p in Int32(1):Int32(N)
            s_p += ops.G[p, p_local] * face_buf[3, p, q_local]
        end
        @inbounds for q in Int32(1):Int32(N)
            s_q += ops.G[q, q_local] * face_buf[4, p_local, q]
        end
        @inbounds üe_loc[i, j, k] += s_p + s_q + face_buf[1, p_local, q_local]
    end
    return nothing
end

################################################################################
# Global Laplacian

"""
    apply_laplacian3d!(ü, u, bdry_values; geom, ops, τ)

Curvilinear-aware 3D discrete Laplacian over the mesh carried by `geom
:: MeshGeometry`. State arrays `u`, `ü` are 4-D of shape `(N, N, N,
Ne)`, with element ordering matching `geom`. Returns `ü = L_h u`.

Boundary conditions: outer faces with `bdry[f, e] ∈ 1..6` are
Dirichlet-driven against `bdry_values[bdry[f, e]]`. Outer faces with
any other tag (including 0) are *free* — the operator contributes
nothing on them, leaving room for the caller to add their own SAT
post-pass (Sommerfeld, Robin, …) on those faces. Interior faces use
the SIPG split.

The kernel assumes a **diagonal** 1D mass matrix (`ops.H` is the GLL
quadrature). For dense-`H` branches (e.g. the Rational/Vandermonde
operator construction) the weak-form assembly would need a different
implementation.

# Choosing τ (SIPG penalty constant)

`τ` must be large enough to make the discrete Laplacian negative
semi-definite. Empirical rules of thumb (N = 5 GLL nodes per element):

* **Axis-aligned cubical mesh**: `τ ≈ 1.5·(N−1)² = 24` is sufficient.
* **Curvilinear / multi-patch mesh** (e.g. `make_cubed_cube_mesh`):
  the SIPG threshold rises significantly because the outer-patch
  elements are anisotropic and skewed. Use `τ ≈ 8·(N−1)² = 128` (for
  N=5). The threshold at which the operator first becomes NSD is around
  `4·(N−1)² ≈ 64`, but staying close to that edge can leave a handful
  of slowly-growing modes which compound over long integrations; the
  extra margin keeps the spectrum cleanly negative.

If the operator's spectrum has positive eigenvalues, evolution will
slowly grow even with a symplectic integrator; you can verify
coercivity for any (geom, ops, τ) tuple via [`discrete_laplacian`](@ref)
on a small representative mesh.
"""
function apply_laplacian3d! end

# ---------- Kernel 1: face-trace gather ------------------------------
#
# Reads `u` and writes `geom.face_trace[1..4, p, q, f, e]`, the four
# scalars `(u, ∂x u, ∂y u, ∂z u)` at each face quadrature node of
# every element. This is the only data each element needs to ship to
# its neighbours.
#
# Workgroup-per-element, N³ workitems/workgroup, one per node. The
# workgroup loads `u` into `@localmem u_loc` so the SBP stencils that
# compute `∇_ref u` at the face nodes can read across workitems.
# Each workitem then checks the six boundary conditions on its
# `(i, j, k)`; for every face it lies on, it computes
# `∇u_phys = J⁻ᵀ · ∇_ref u` and writes one entry of `face_trace`. A
# corner workitem (on three faces) writes three entries; interior
# workitems (off every face) compute nothing — at N=4 that's the
# central 2×2×2 = 8 of the 64 workitems.
#
# We deliberately *do not* compute the volume divergence here: kernel
# 2 below recomputes the gradient from the same `u`. The redundancy
# (one extra global-memory `u` load + one extra stencil per element)
# is small relative to the work saved by not having to round-trip
# `Du_loc` or `ü_pre` through global memory between kernels.
#
# `ndrange = N³·Ne`, `workgroupsize = N³`.
@kernel function _laplacian3d_face_trace_kernel!(@Const(u::AbstractArray{T}),
                                            geom, ops, ::Val{N}) where {T, N}
    e        = @index(Group, Linear)
    li       = @index(Local, Linear)
    i, j, k  = _ijk_from_li(li, Val(N))

    u_loc = @localmem T (N, N, N)

    # Load u into shared.
    @inbounds u_loc[i, j, k] = u[i, j, k, e]
    @synchronize

    # Re-derive indices after the barrier (KA CPU semantics).
    e        = @index(Group, Linear)
    li       = @index(Local, Linear)
    i, j, k  = _ijk_from_li(li, Val(N))

    # Skip workitems not on any face — saves the gradient compute for
    # interior nodes (at N=4 that's 8 of the 64 workitems; at N=8 it
    # would be ~75% — more meaningful savings).
    on_face = (i == 1) | (i == N) | (j == 1) | (j == N) | (k == 1) | (k == N)
    if on_face
        @inbounds begin
            # Reference gradient at this workitem's own collocation
            # point. The same value will be written to each face_trace
            # slot the workitem is responsible for.
            s1 = zero(T); s2 = zero(T); s3 = zero(T)
            for l in Int32(1):Int32(N)
                s1 += ops.G[i, l] * u_loc[l, j, k]
                s2 += ops.G[j, l] * u_loc[i, l, k]
                s3 += ops.G[k, l] * u_loc[i, j, l]
            end

            # Physical gradient `∇u_phys = J⁻ᵀ · ∇_ref u`.
            Ji11 = geom.invjac[1,1,i,j,k,e]; Ji12 = geom.invjac[1,2,i,j,k,e]; Ji13 = geom.invjac[1,3,i,j,k,e]
            Ji21 = geom.invjac[2,1,i,j,k,e]; Ji22 = geom.invjac[2,2,i,j,k,e]; Ji23 = geom.invjac[2,3,i,j,k,e]
            Ji31 = geom.invjac[3,1,i,j,k,e]; Ji32 = geom.invjac[3,2,i,j,k,e]; Ji33 = geom.invjac[3,3,i,j,k,e]
            gx = Ji11*s1 + Ji21*s2 + Ji31*s3
            gy = Ji12*s1 + Ji22*s2 + Ji32*s3
            gz = Ji13*s1 + Ji23*s2 + Ji33*s3
            u_val = u_loc[i, j, k]

            # Write trace entry for every face this workitem is on.
            # `(p_local, q_local)` follows the per-face axis convention
            # documented in the "Face axis bookkeeping" block above.
            if i == 1
                geom.face_trace[1, j, k, 1, e] = u_val
                geom.face_trace[2, j, k, 1, e] = gx
                geom.face_trace[3, j, k, 1, e] = gy
                geom.face_trace[4, j, k, 1, e] = gz
            end
            if i == N
                geom.face_trace[1, j, k, 2, e] = u_val
                geom.face_trace[2, j, k, 2, e] = gx
                geom.face_trace[3, j, k, 2, e] = gy
                geom.face_trace[4, j, k, 2, e] = gz
            end
            if j == 1
                geom.face_trace[1, i, k, 3, e] = u_val
                geom.face_trace[2, i, k, 3, e] = gx
                geom.face_trace[3, i, k, 3, e] = gy
                geom.face_trace[4, i, k, 3, e] = gz
            end
            if j == N
                geom.face_trace[1, i, k, 4, e] = u_val
                geom.face_trace[2, i, k, 4, e] = gx
                geom.face_trace[3, i, k, 4, e] = gy
                geom.face_trace[4, i, k, 4, e] = gz
            end
            if k == 1
                geom.face_trace[1, i, j, 5, e] = u_val
                geom.face_trace[2, i, j, 5, e] = gx
                geom.face_trace[3, i, j, 5, e] = gy
                geom.face_trace[4, i, j, 5, e] = gz
            end
            if k == N
                geom.face_trace[1, i, j, 6, e] = u_val
                geom.face_trace[2, i, j, 6, e] = gx
                geom.face_trace[3, i, j, 6, e] = gy
                geom.face_trace[4, i, j, 6, e] = gz
            end
        end
    end
end

# ---------- Kernel 2: volume work + face SAT + mass division --------
#
# Reads `u` (again) and `geom.face_trace` (written by kernel 1) and
# computes the full right-hand side `ü`. The phases inside the
# workgroup are:
#
#   1. Load u → u_loc. @synchronize.
#   2. Compute reference gradients (registers) + weak stiffness flux
#      W_loc (shared) at each node. @synchronize.
#   3. Volume divergence `−Σ_a Gᵀ_a W_a` → üe_loc (shared, pre-mass-
#      div). No barrier needed before face SAT because the divergence
#      writes only `üe_loc[i, j, k]` (the workitem's own slot) and
#      face SAT only reads it from the same slot.
#   4. Six face SAT phases. For each face:
#        a. `_face_sat_compute!` — on-face workitems pack the four
#           SAT scalars (`bcorr`, `μ_a·ν`, `μ_axis_p·ν`,
#           `μ_axis_q·ν`) into `face_buf`. Off-face workitems return.
#        @synchronize so all workitems see the freshly-written
#        `face_buf`.
#        b. `_face_sat_apply!` — every workitem adds its share to
#           `üe_loc[i, j, k]`. The "interior" lift hits everyone; the
#           in-face lifts + boundary correction only fire when the
#           workitem itself is on the face.
#        @synchronize so `face_buf` is safe to overwrite for the
#        next face.
#   5. Mass division + writeback. Each workitem divides its own
#      `üe_loc[i, j, k]` by `Hphys` and writes the result to `ü`.
#
# 12 barriers total inside the face SAT (2 per face), plus the 2
# inside the volume phase. Each barrier is ~ns on CUDA/Metal so the
# absolute cost is negligible at production scale.
@kernel function _laplacian3d_volume_kernel!(ü::AbstractArray{T},
                                              @Const(u::AbstractArray{T}),
                                              geom, ops, τ::T,
                                              bdry_values, H_1d,
                                              ::Val{N}) where {T, N}
    e        = @index(Group, Linear)
    li       = @index(Local, Linear)
    i, j, k  = _ijk_from_li(li, Val(N))

    # Workgroup-local scratch.
    u_loc    = @localmem T (N, N, N)         # field values
    W_loc    = @localmem T (3, N, N, N)      # weak stiffness flux
    üe_loc   = @localmem T (N, N, N)         # accumulating L·u (pre-mass-div)
    face_buf = @localmem T (4, N, N)         # per-face SAT scratch, reused

    # Load u into shared.
    @inbounds u_loc[i, j, k] = u[i, j, k, e]
    @synchronize

    # ---- Volume work: gradient → flux → divergence into üe_loc ----
    e       = @index(Group, Linear)
    li      = @index(Local, Linear)
    i, j, k = _ijk_from_li(li, Val(N))

    s1 = zero(T); s2 = zero(T); s3 = zero(T)
    @inbounds for l in Int32(1):Int32(N)
        s1 += ops.G[i, l] * u_loc[l, j, k]
        s2 += ops.G[j, l] * u_loc[i, l, k]
        s3 += ops.G[k, l] * u_loc[i, j, l]
    end

    # Step 2: weak stiffness flux (per-node, no cross-workitem reads).
    @inbounds begin
        Ji11 = geom.invjac[1,1,i,j,k,e]; Ji12 = geom.invjac[1,2,i,j,k,e]; Ji13 = geom.invjac[1,3,i,j,k,e]
        Ji21 = geom.invjac[2,1,i,j,k,e]; Ji22 = geom.invjac[2,2,i,j,k,e]; Ji23 = geom.invjac[2,3,i,j,k,e]
        Ji31 = geom.invjac[3,1,i,j,k,e]; Ji32 = geom.invjac[3,2,i,j,k,e]; Ji33 = geom.invjac[3,3,i,j,k,e]
        wt   = geom.Hphys[i, j, k, e]

        g11 = Ji11*Ji11 + Ji12*Ji12 + Ji13*Ji13
        g12 = Ji11*Ji21 + Ji12*Ji22 + Ji13*Ji23
        g13 = Ji11*Ji31 + Ji12*Ji32 + Ji13*Ji33
        g22 = Ji21*Ji21 + Ji22*Ji22 + Ji23*Ji23
        g23 = Ji21*Ji31 + Ji22*Ji32 + Ji23*Ji33
        g33 = Ji31*Ji31 + Ji32*Ji32 + Ji33*Ji33

        W_loc[1, i, j, k] = wt * (g11*s1 + g12*s2 + g13*s3)
        W_loc[2, i, j, k] = wt * (g12*s1 + g22*s2 + g23*s3)
        W_loc[3, i, j, k] = wt * (g13*s1 + g23*s2 + g33*s3)
    end
    @synchronize

    # Step 3: volume divergence → üe_loc (kept in shared throughout).
    e       = @index(Group, Linear)
    li      = @index(Local, Linear)
    i, j, k = _ijk_from_li(li, Val(N))
    s = zero(T)
    @inbounds for l in Int32(1):Int32(N)
        s += ops.G[l, i] * W_loc[1, l, j, k]
        s += ops.G[l, j] * W_loc[2, i, l, k]
        s += ops.G[l, k] * W_loc[3, i, j, l]
    end
    @inbounds üe_loc[i, j, k] = -s

    # ---- Face SAT: 6 faces, each compute + apply, with barriers ----

    # Face 1: compute, sync, apply, sync.
    _face_sat_compute!(Val(1), face_buf, i, j, k, e, geom, geom.conn, ops, τ, bdry_values, H_1d, Val(N))
    @synchronize
    e       = @index(Group, Linear); li = @index(Local, Linear)
    i, j, k = _ijk_from_li(li, Val(N))
    _face_sat_apply!(Val(1), üe_loc, face_buf, i, j, k, ops, Val(N))
    @synchronize

    # Face 2.
    e       = @index(Group, Linear); li = @index(Local, Linear)
    i, j, k = _ijk_from_li(li, Val(N))
    _face_sat_compute!(Val(2), face_buf, i, j, k, e, geom, geom.conn, ops, τ, bdry_values, H_1d, Val(N))
    @synchronize
    e       = @index(Group, Linear); li = @index(Local, Linear)
    i, j, k = _ijk_from_li(li, Val(N))
    _face_sat_apply!(Val(2), üe_loc, face_buf, i, j, k, ops, Val(N))
    @synchronize

    # Face 3.
    e       = @index(Group, Linear); li = @index(Local, Linear)
    i, j, k = _ijk_from_li(li, Val(N))
    _face_sat_compute!(Val(3), face_buf, i, j, k, e, geom, geom.conn, ops, τ, bdry_values, H_1d, Val(N))
    @synchronize
    e       = @index(Group, Linear); li = @index(Local, Linear)
    i, j, k = _ijk_from_li(li, Val(N))
    _face_sat_apply!(Val(3), üe_loc, face_buf, i, j, k, ops, Val(N))
    @synchronize

    # Face 4.
    e       = @index(Group, Linear); li = @index(Local, Linear)
    i, j, k = _ijk_from_li(li, Val(N))
    _face_sat_compute!(Val(4), face_buf, i, j, k, e, geom, geom.conn, ops, τ, bdry_values, H_1d, Val(N))
    @synchronize
    e       = @index(Group, Linear); li = @index(Local, Linear)
    i, j, k = _ijk_from_li(li, Val(N))
    _face_sat_apply!(Val(4), üe_loc, face_buf, i, j, k, ops, Val(N))
    @synchronize

    # Face 5.
    e       = @index(Group, Linear); li = @index(Local, Linear)
    i, j, k = _ijk_from_li(li, Val(N))
    _face_sat_compute!(Val(5), face_buf, i, j, k, e, geom, geom.conn, ops, τ, bdry_values, H_1d, Val(N))
    @synchronize
    e       = @index(Group, Linear); li = @index(Local, Linear)
    i, j, k = _ijk_from_li(li, Val(N))
    _face_sat_apply!(Val(5), üe_loc, face_buf, i, j, k, ops, Val(N))
    @synchronize

    # Face 6.
    e       = @index(Group, Linear); li = @index(Local, Linear)
    i, j, k = _ijk_from_li(li, Val(N))
    _face_sat_compute!(Val(6), face_buf, i, j, k, e, geom, geom.conn, ops, τ, bdry_values, H_1d, Val(N))
    @synchronize
    e       = @index(Group, Linear); li = @index(Local, Linear)
    i, j, k = _ijk_from_li(li, Val(N))
    _face_sat_apply!(Val(6), üe_loc, face_buf, i, j, k, ops, Val(N))
    @synchronize

    # ---- Mass division + global writeback ----
    e       = @index(Group, Linear); li = @index(Local, Linear)
    i, j, k = _ijk_from_li(li, Val(N))
    @inbounds üe_loc[i, j, k] /= geom.Hphys[i, j, k, e]
    @inbounds ü[i, j, k, e] = üe_loc[i, j, k]
end

function apply_laplacian3d!(ü::AbstractArray{T,4}, u::AbstractArray{T,4},
                             bdry_values::NTuple{6, T};
                             geom::MeshGeometry{T, N}, ops::SBPOps{N, T}, τ) where {N, T}
    @assert size(ü) == size(u)
    @assert size(u, 1) == size(u, 2) == size(u, 3) == N
    @assert size(u, 4) == geom.Ne

    H_1d = SVector{N, T}(ntuple(i -> ops.H[i, i], Val(N)))
    backend = get_backend(u)

    # Two-launch design. Kernel 1 only reads `u` and writes
    # `geom.face_trace`. Kernel 2 reads `u` again *and*
    # `geom.face_trace`, then writes `ü`. KA's per-backend command
    # queue is FIFO, so kernel 2 sees kernel 1's writes without an
    # explicit `synchronize`. Both kernels use the same workgroup-
    # per-element layout (N³ workitems/workgroup, one per node).
    _laplacian3d_face_trace_kernel!(backend, N^3)(
        u, geom, ops, Val(N);
        ndrange = N^3 * geom.Ne)
    _laplacian3d_volume_kernel!(backend, N^3)(
        ü, u, geom, ops, T(τ), bdry_values, H_1d, Val(N);
        ndrange = N^3 * geom.Ne)

    return ü
end

################################################################################
# Diagnostics

using SparseArrays: SparseMatrixCSC, sparse

"""
    discrete_laplacian(geom, ops, τ;
                       bdry_values = ntuple(_->0, 6),
                       drop_tol    = 0) → SparseMatrixCSC

Assemble the curvilinear discrete Laplacian `L_h` as an explicit sparse
matrix by repeatedly calling `apply_laplacian3d!` on each canonical
basis vector. The result `L_h` is the matrix that maps the linearly-
indexed global state vector `vec(u)` to `vec(apply_laplacian3d!(u))`
(with zero boundary data, so only the linear part of the homogeneous
Laplacian is captured).

Intended for analysis — eigenvalues, symmetry, condition number — not
for production timestepping. Cost is `O(N⁶ · Ne²)` element operations,
i.e. one `apply_laplacian3d!` call per degree of freedom.

**Host-only.** The per-column basis-vector trick uses scalar indexing
on a flat host vector; if `geom` is device-resident, migrate it back
with a fresh `make_geometry(mesh, elem)` first.

The per-node physical mass `H_phys` (which makes `H_phys · L_h` the
symmetric "stiffness" matrix) is available as
[`physical_mass_diagonal`](@ref).

# Keyword arguments

* `bdry_values` — passed straight through to `apply_laplacian3d!`;
  defaults to zero Dirichlet on every outer-face tag.
* `drop_tol` — entries with `abs(v) ≤ drop_tol` are not stored. Default
  `0` keeps every numerically computed value.
"""
function discrete_laplacian(geom::MeshGeometry{T, N}, ops::SBPOps{N, T}, τ;
                            bdry_values::NTuple{6, T} = ntuple(_ -> zero(T), Val(6)),
                            drop_tol = zero(T)) where {N, T}
    Ne   = geom.Ne
    ndof = N^3 * Ne
    u  = zeros(T, N, N, N, Ne)
    ü = similar(u)

    I_idx = Int[];  J_idx = Int[];  V = T[]
    sizehint!(I_idx, 64 * ndof);  sizehint!(J_idx, 64 * ndof);  sizehint!(V, 64 * ndof)

    u_flat = vec(u)
    ü_flat = vec(ü)
    @inbounds for col in 1:ndof
        u_flat[col] = one(T)
        apply_laplacian3d!(ü, u, bdry_values; geom, ops, τ)
        u_flat[col] = zero(T)
        for row in 1:ndof
            v = ü_flat[row]
            if abs(v) > drop_tol
                push!(I_idx, row);  push!(J_idx, col);  push!(V, v)
            end
        end
    end
    return sparse(I_idx, J_idx, V, ndof, ndof)
end

"""
    spectral_radius_estimate(geom, ops, τ;
                             tol = 1e-4, maxiter = 50, krylovdim = 30) → T

Estimate `|λ_max|`, the magnitude of the largest eigenvalue of the
curvilinear discrete Laplacian, by running `KrylovKit.eigsolve(:LM)`
on a matrix-free shell that wraps [`apply_laplacian3d!`](@ref). For a
negative-semi-definite operator this equals `|λ_min|`, which the wave-
equation timestep uses as `ω_max²`. Other equations consume the same
`|λ_max|` differently.

The Lanczos algorithm (forced via `issymmetric = true`) converges in
~3–5 iterations of `krylovdim` matvecs in practice. The matvec is
`apply_laplacian3d!` itself, so this runs on whatever
`KernelAbstractions` backend `geom.coords` lives on (CPU, Metal,
CUDA, …).
"""
function spectral_radius_estimate(geom::MeshGeometry{T, N}, ops::SBPOps{N, T}, τ;
                                    tol = T(1e-4),
                                    maxiter::Int = 50,
                                    krylovdim::Int = 30) where {N, T}
    Ne   = geom.Ne
    bdry = ntuple(_ -> zero(T), Val(6))

    backend = get_backend(geom.coords)

    # Random starting vector (same backend as geom).
    x₀ = KernelAbstractions.allocate(backend, T, N, N, N, Ne)
    Random.randn!(x₀)
    n0 = sqrt(sum(abs2, x₀))
    n0 == 0 && return zero(T)
    x₀ ./= n0

    # Matrix-free shell: `y = L_h · x`. `eigsolve` calls this many
    # times; the per-call allocation of `y` is dominated by the
    # operator work, so out-of-place is fine.
    apply_Lh = let geom = geom, ops = ops, τ = τ, bdry = bdry
        x -> begin
            y = similar(x)
            apply_laplacian3d!(y, x, bdry; geom, ops, τ)
            return y
        end
    end

    vals, _, _ = KrylovKit.eigsolve(apply_Lh, x₀, 1, :LM;
                                     issymmetric = true,
                                     tol = tol,
                                     maxiter = maxiter,
                                     krylovdim = krylovdim)
    return abs(vals[1])
end

"""
    discrete_inner_product(u, v, geom, ops) → T

Discrete physical inner product `⟨u, v⟩_{H_phys} = Σ H_phys · u · v`,
where `H_phys[i, j, k, e] = H_ref[i] H_ref[j] H_ref[k] · |det J|` is the
per-node mass induced by the GLL quadrature on the curvilinear element
map. This is the natural inner product on the SBP-DG discretisation —
it recovers the continuous `L²(Ω)` inner product in the high-resolution
limit and makes the discrete Laplacian
(operator from `apply_laplacian3d!`) symmetric in this inner product on
a coercive mesh.

`u` and `v` are 4-D arrays of shape `(N, N, N, Ne)`.
"""
function discrete_inner_product(u::AbstractArray{T, 4}, v::AbstractArray{T, 4},
                                 geom::MeshGeometry{T, N},
                                 ops::SBPOps{N, T}) where {N, T}
    @assert size(u) == size(v) == size(geom.Hphys) == (N, N, N, geom.Ne)
    # Three-array `mapreduce` — works for plain `Array` on CPU and for
    # `MtlArray` / `CuArray` / `ROCArray` on their respective backends
    # via the GPUArrays.jl broadcast-backed `mapreducedim!`. The per-
    # node physical mass `geom.Hphys` is precomputed at `make_geometry`
    # so no auxiliary closure or capture is needed here. `ops` is
    # accepted only to preserve the public signature; the mass diagonal
    # already lives in `geom`.
    return mapreduce((uᵢ, vᵢ, hᵢ) -> uᵢ * vᵢ * hᵢ, +, u, v, geom.Hphys;
                     init = zero(T))
end

"""
    discrete_l2_norm(u, geom, ops) → T

Physical-mass-weighted L² norm `sqrt(⟨u, u⟩_{H_phys})`. See
[`discrete_inner_product`](@ref).
"""
discrete_l2_norm(u::AbstractArray{T, 4}, geom::MeshGeometry{T, N},
                  ops::SBPOps{N, T}) where {N, T} =
    sqrt(discrete_inner_product(u, u, geom, ops))

"""
    physical_mass_diagonal(geom, ops) → Vector{T}

Diagonal of the global mass matrix `H_phys`, in the same linear node
ordering as `vec(u)`. Useful in conjunction with [`discrete_laplacian`](@ref)
to form `H_phys · L_h` or to solve the generalized eigenproblem
`L_h · v = λ · v` against the physical inner product.

Host-only analysis tool. The result is always a CPU `Vector{T}`; if
`geom.Hphys` is device-resident it is brought back to host. `ops` is
accepted for signature compatibility but no longer consulted — the
per-node physical mass lives in `geom.Hphys` since `make_geometry`.
"""
physical_mass_diagonal(geom::MeshGeometry{T, N}, ops::SBPOps{N, T}) where {N, T} =
    vec(Array(geom.Hphys))
