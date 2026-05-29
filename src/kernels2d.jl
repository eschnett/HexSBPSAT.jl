# Curvilinear-aware 2D discrete Laplacian on a `MeshGeometry` (built on
# top of a 2D `Mesh` from `HexMeshes`). Mirror of `kernels3d.jl`
# adapted to two spatial dimensions:
#
#   * 4 faces per element (vs 6 in 3D), labelled
#     1 = −x, 2 = +x, 3 = −y, 4 = +y.
#   * Each face is a 1-D edge with `N` quadrature nodes (vs an N×N
#     quadrature patch in 3D).
#   * Face-orientation group is D₁ (2 elements) — encoded by
#     `o ∈ {0, 1}` and resolved by `HexMeshes._neigh_p`.
#   * Outward physical normal is the 90° rotation of the in-face tangent
#     (instead of the 3D cross product of two tangents).
#
# Public entry point: `apply_laplacian!(ü, u, bdry_values; geom, ops, τ)`.
# Equation-agnostic, exactly like the 3D version — downstream packages
# layer wave-equation BCs (Sommerfeld, Robin, …) on top by post-passing
# over outer faces.

using KernelAbstractions: @kernel, @index, @Const,
                          get_backend, synchronize

################################################################################
# Face axis bookkeeping (2D)
#
# Face index convention (matches `Mesh{2}`):
#   1 = −x    2 = +x    3 = −y    4 = +y
#
# For face `f`, the orthogonal reference axis is `a = (f + 1) ÷ 2` and
# the (single) in-face tangent reference axis is `axis_p = 3 − a`. The
# face quadrature node index `p` maps to the volume nodes as:
#
#   a = 1 → axis_p = 2,  volume node = (face_row, p)
#   a = 2 → axis_p = 1,  volume node = (p, face_row)
#
# `face_row = 1` on a −face and `N` on a +face. The outward physical
# normal is the in-plane perpendicular of the tangent `t = J[:, axis_p]`,
# i.e. `n_u = sgn_out · (t_y, −t_x)` with `sgn_out = sgn_f · sgn_c ·
# handedness[e]` and `sgn_c = +1 if a == 1 else −1`. For a unit-square
# element with identity Jacobian this recovers the obvious outward
# normals `(±1, 0)` / `(0, ±1)`.

# Per-kernel index typing: matches the conventions in `kernels3d.jl` —
# small constants returned as `Int` (so they can serve as `Val{…}`
# parameters), workitem indices kept in `Int32` so the per-element
# address arithmetic stays off the 64-bit slow path on NVIDIA and Apple
# Silicon GPUs.
@inline _face_axis_idx_2d(::Val{f}) where {f} = (f + 1) ÷ 2
@inline _face_row_2d(::Val{f}, ::Val{N}) where {f, N} = isodd(f) ? 1 : N
@inline _face_sign_2d(::Val{f}, ::Type{T}) where {f, T} = isodd(f) ? -one(T) : one(T)
@inline _cross_sign_2d(::Val{a}, ::Type{T}) where {a, T} = a == 1 ? one(T) : -one(T)
@inline _tangent_axis_2d(::Val{1}) = Int32(2)
@inline _tangent_axis_2d(::Val{2}) = Int32(1)

# Volume index of face node `p` for face axis `a` and `face_row`.
@inline _face_volume_idx_2d(::Val{1}, face_row, p) = (face_row, p)
@inline _face_volume_idx_2d(::Val{2}, face_row, p) = (p, face_row)

# Decode workitem-local `(i, j) ∈ Int32(1):Int32(N)` from KA's
# `@index(Local, Linear)`. Counterpart of `_ijk_from_li` in
# `kernels3d.jl`; called fresh after every `@synchronize` because CPU-
# backend locals don't survive a barrier.
@inline function _ij_from_li(li, ::Val{N}) where {N}
    li0 = (li % Int32) - Int32(1)
    n   = Int32(N)
    i   = (li0 % n) + Int32(1)
    j   = (li0 ÷ n) + Int32(1)
    return i, j
end

################################################################################
# Face-trace gather (pass 1) + per-face SAT (pass 2)
#
# Two-launch design mirroring `kernels3d.jl`. Pass 1 reads `u` and
# writes `work.face_trace[1..3, p, f, e]` — `(u, ∂xu, ∂yu)` at every
# face quadrature node. Pass 2 reads `u` and `face_trace`, applies the
# volume divergence + the four face SATs + the mass division, and
# writes `ü`. KA's per-backend FIFO command queue makes pass 2 see
# pass 1's writes without an explicit `synchronize`.

# ---- Pass 2: face SAT helpers (compute / apply) ---------------------
#
# `_face_sat_compute_2d!`: only the N workitems whose `(i, j)` lies on
# the face are active. Each one writes its share of the SAT to
# `face_buf[1..3, p_local]`:
#   slot 1 — `bcorr = wF · (Gn_self − ½ ΔGn − τ Δu / hF)` (boundary
#            lift + adjoint consistency + SIPG penalty);
#   slot 2 — `μ_a · ν`, weight for the "interior" SIPG lift along the
#            face's orthogonal axis;
#   slot 3 — `μ_axis_p · ν`, weight for the "in-face" SIPG lift along
#            the face's tangent axis.
@inline function _face_sat_compute_2d!(::Val{f}, face_buf,
                                       i, j, e,
                                       geom::MeshGeometry{2, T, N},
                                       work::MeshWorkspace{2, T, N},
                                       conn::MeshConnectivity,
                                       ops::SBPOps{N, T}, τ::T,
                                       bdry_values::NTuple{4, T},
                                       H_1d::SVector{N, T},
                                       ::Val{N}) where {f, T, N}
    a_idx  = _face_axis_idx_2d(Val(f))
    face_r = _face_row_2d(Val(f), Val(N))
    sgn_f  = _face_sign_2d(Val(f), T)
    sgn_c  = _cross_sign_2d(Val(a_idx), T)
    axis_p = _tangent_axis_2d(Val(a_idx))

    ia = a_idx == 1 ? i : j
    ia == face_r || return nothing

    p_local = a_idx == 1 ? j : i

    @inbounds J11 = geom.jac[1,1,i,j,e]; @inbounds J12 = geom.jac[1,2,i,j,e]
    @inbounds J21 = geom.jac[2,1,i,j,e]; @inbounds J22 = geom.jac[2,2,i,j,e]

    tp_x = axis_p == 1 ? J11 : J12
    tp_y = axis_p == 1 ? J21 : J22

    @inbounds sgn_out = sgn_f * sgn_c * T(geom.handedness[e])

    # 90° rotation of the tangent gives the outward physical normal
    # (unnormalised): `n_u = sgn_out · (t_y, −t_x)`. The magnitude
    # `JF = |n_u|` is the surface element (= edge length scaling).
    nx_u =  sgn_out * tp_y
    ny_u = -sgn_out * tp_x
    JF   = sqrt(nx_u * nx_u + ny_u * ny_u)
    nx   = nx_u / JF
    ny   = ny_u / JF
    # SIPG penalty length: perpendicular element thickness
    # `hF = |K_e| / |F| = detjac / JF` at this face node. Matches the
    # 3D code (which fixed a stretched-element under-penalisation bug
    # against the geometric-mean `sqrt(JF)`).
    @inbounds dJ = geom.detjac[i, j, e]
    hF = dJ / JF

    @inbounds Ji11 = geom.invjac[1,1,i,j,e]; @inbounds Ji12 = geom.invjac[1,2,i,j,e]
    @inbounds Ji21 = geom.invjac[2,1,i,j,e]; @inbounds Ji22 = geom.invjac[2,2,i,j,e]
    μ1 = Ji11 * nx + Ji12 * ny
    μ2 = Ji21 * nx + Ji22 * ny

    @inbounds wF = JF * H_1d[p_local]

    @inbounds u_self  = work.face_trace[1, p_local, f, e]
    @inbounds gx_self = work.face_trace[2, p_local, f, e]
    @inbounds gy_self = work.face_trace[3, p_local, f, e]
    Gn_self = nx * gx_self + ny * gy_self

    @inbounds nbr = conn.neighbour[f, e]
    @inbounds tag = conn.bdry[f, e]
    α = nbr == 0 ? one(T) : one(T) / 2

    Δu::T  = zero(T)
    ΔGn::T = zero(T)
    if nbr == 0
        # Outer face. Tags 1..4 → Dirichlet drive against
        # `bdry_values[tag]`. Every other tag (incl. 0) is "free":
        # contributes only the natural boundary lift `wF · Gn_self`.
        # The downstream caller layers its own BC (Sommerfeld, Robin,
        # Neumann, …) on top by post-passing.
        if Int8(1) ≤ tag ≤ Int8(4)
            @inbounds u_neigh = bdry_values[tag]
            Δu = u_self - u_neigh
        end
    else
        @inbounds nbr_face = Int32(conn.neighbour_face[f, e])
        @inbounds nbr_o    = conn.orientation[f, e]
        pn = _neigh_p(nbr_o, p_local, Int32(N))
        @inbounds u_neigh = work.face_trace[1, pn, nbr_face, nbr]
        @inbounds gx_n    = work.face_trace[2, pn, nbr_face, nbr]
        @inbounds gy_n    = work.face_trace[3, pn, nbr_face, nbr]
        Gn_neigh = nx * gx_n + ny * gy_n
        Δu  = u_self  - u_neigh
        ΔGn = Gn_self - Gn_neigh
    end

    ν = α * wF * Δu

    μa = a_idx  == 1 ? μ1 : μ2
    μp = axis_p == 1 ? μ1 : μ2

    bcorr = wF * (Gn_self - (T(1) / 2) * ΔGn - τ * Δu / hF)

    @inbounds face_buf[1, p_local] = bcorr
    @inbounds face_buf[2, p_local] = μa * ν
    @inbounds face_buf[3, p_local] = μp * ν
    return nothing
end

@inline function _face_sat_apply_2d!(::Val{f}, üe_loc, face_buf,
                                     i, j,
                                     ops::SBPOps{N, T},
                                     ::Val{N}) where {f, T, N}
    a_idx  = _face_axis_idx_2d(Val(f))
    face_r = _face_row_2d(Val(f), Val(N))

    ia = a_idx == 1 ? i : j
    p_local = a_idx == 1 ? j : i

    # Interior lift along the face's orthogonal axis. Spreads
    # `μ_a · ν` along an axis-a line through the volume; every
    # workitem receives a contribution weighted by `G[face_r, ia]`.
    @inbounds üe_loc[i, j] += ops.G[face_r, ia] * face_buf[2, p_local]

    # In-face lift + boundary correction. Only fires on workitems that
    # are themselves on the face. The gather replaces what would
    # otherwise be a scatter (`üe[i, l] += G[j, l] · μ · ν`).
    if ia == face_r
        s_p = zero(T)
        @inbounds for p in Int32(1):Int32(N)
            s_p += ops.G[p, p_local] * face_buf[3, p]
        end
        @inbounds üe_loc[i, j] += s_p + face_buf[1, p_local]
    end
    return nothing
end

################################################################################
# Global Laplacian (2D)
#
# The user-visible docstring for `apply_laplacian!` lives on the 3D
# method in `kernels3d.jl` and explicitly notes the 2D dispatch.
# `bdry_values :: NTuple{4, T}` supplies Dirichlet values for outer
# faces tagged `1..4`; any other tag is a "free" face that contributes
# only the natural boundary lift, matching the 3D semantics.

# ---------- Kernel 1: face-trace gather (2D) ------------------------
@kernel function _laplacian2d_face_trace_kernel!(@Const(u::AbstractArray{T}),
                                            geom, work, ops, ::Val{N}) where {T, N}
    e       = @index(Group, Linear)
    li      = @index(Local, Linear)
    i, j    = _ij_from_li(li, Val(N))

    u_loc = @localmem T (N, N)

    @inbounds u_loc[i, j] = u[i, j, e]
    @synchronize

    e    = @index(Group, Linear)
    li   = @index(Local, Linear)
    i, j = _ij_from_li(li, Val(N))

    on_face = (i == 1) | (i == N) | (j == 1) | (j == N)
    if on_face
        @inbounds begin
            s1 = zero(T); s2 = zero(T)
            for l in Int32(1):Int32(N)
                s1 += ops.G[i, l] * u_loc[l, j]
                s2 += ops.G[j, l] * u_loc[i, l]
            end

            # Physical gradient `∇u_phys = J⁻ᵀ · ∇_ref u`.
            Ji11 = geom.invjac[1,1,i,j,e]; Ji12 = geom.invjac[1,2,i,j,e]
            Ji21 = geom.invjac[2,1,i,j,e]; Ji22 = geom.invjac[2,2,i,j,e]
            gx = Ji11 * s1 + Ji21 * s2
            gy = Ji12 * s1 + Ji22 * s2
            u_val = u_loc[i, j]

            # Per-face trace writes — `p_local` follows the convention
            # documented at the top of this file.
            if i == 1
                work.face_trace[1, j, 1, e] = u_val
                work.face_trace[2, j, 1, e] = gx
                work.face_trace[3, j, 1, e] = gy
            end
            if i == N
                work.face_trace[1, j, 2, e] = u_val
                work.face_trace[2, j, 2, e] = gx
                work.face_trace[3, j, 2, e] = gy
            end
            if j == 1
                work.face_trace[1, i, 3, e] = u_val
                work.face_trace[2, i, 3, e] = gx
                work.face_trace[3, i, 3, e] = gy
            end
            if j == N
                work.face_trace[1, i, 4, e] = u_val
                work.face_trace[2, i, 4, e] = gx
                work.face_trace[3, i, 4, e] = gy
            end
        end
    end
end

# ---------- Kernel 2: volume work + face SAT + mass division (2D) ---
@kernel function _laplacian2d_volume_kernel!(ü::AbstractArray{T},
                                              @Const(u::AbstractArray{T}),
                                              geom, work, ops, τ::T,
                                              bdry_values, H_1d,
                                              ::Val{N}) where {T, N}
    e    = @index(Group, Linear)
    li   = @index(Local, Linear)
    i, j = _ij_from_li(li, Val(N))

    u_loc    = @localmem T (N, N)
    W_loc    = @localmem T (2, N, N)
    üe_loc   = @localmem T (N, N)
    face_buf = @localmem T (3, N)

    @inbounds u_loc[i, j] = u[i, j, e]
    @synchronize

    # ---- Volume work: gradient → flux → divergence ----
    e    = @index(Group, Linear)
    li   = @index(Local, Linear)
    i, j = _ij_from_li(li, Val(N))

    s1 = zero(T); s2 = zero(T)
    @inbounds for l in Int32(1):Int32(N)
        s1 += ops.G[i, l] * u_loc[l, j]
        s2 += ops.G[j, l] * u_loc[i, l]
    end

    @inbounds begin
        Ji11 = geom.invjac[1,1,i,j,e]; Ji12 = geom.invjac[1,2,i,j,e]
        Ji21 = geom.invjac[2,1,i,j,e]; Ji22 = geom.invjac[2,2,i,j,e]
        wt   = geom.Hphys[i, j, e]

        # Contravariant metric `g^ab = (J⁻¹)·(J⁻¹)^T = J⁻¹ J⁻ᵀ`.
        g11 = Ji11 * Ji11 + Ji12 * Ji12
        g12 = Ji11 * Ji21 + Ji12 * Ji22
        g22 = Ji21 * Ji21 + Ji22 * Ji22

        W_loc[1, i, j] = wt * (g11 * s1 + g12 * s2)
        W_loc[2, i, j] = wt * (g12 * s1 + g22 * s2)
    end
    @synchronize

    e    = @index(Group, Linear)
    li   = @index(Local, Linear)
    i, j = _ij_from_li(li, Val(N))
    s = zero(T)
    @inbounds for l in Int32(1):Int32(N)
        s += ops.G[l, i] * W_loc[1, l, j]
        s += ops.G[l, j] * W_loc[2, i, l]
    end
    @inbounds üe_loc[i, j] = -s

    # ---- Face SAT: 4 faces × (compute + apply) ----

    _face_sat_compute_2d!(Val(1), face_buf, i, j, e, geom, work, geom.conn, ops, τ, bdry_values, H_1d, Val(N))
    @synchronize
    e    = @index(Group, Linear); li = @index(Local, Linear)
    i, j = _ij_from_li(li, Val(N))
    _face_sat_apply_2d!(Val(1), üe_loc, face_buf, i, j, ops, Val(N))
    @synchronize

    e    = @index(Group, Linear); li = @index(Local, Linear)
    i, j = _ij_from_li(li, Val(N))
    _face_sat_compute_2d!(Val(2), face_buf, i, j, e, geom, work, geom.conn, ops, τ, bdry_values, H_1d, Val(N))
    @synchronize
    e    = @index(Group, Linear); li = @index(Local, Linear)
    i, j = _ij_from_li(li, Val(N))
    _face_sat_apply_2d!(Val(2), üe_loc, face_buf, i, j, ops, Val(N))
    @synchronize

    e    = @index(Group, Linear); li = @index(Local, Linear)
    i, j = _ij_from_li(li, Val(N))
    _face_sat_compute_2d!(Val(3), face_buf, i, j, e, geom, work, geom.conn, ops, τ, bdry_values, H_1d, Val(N))
    @synchronize
    e    = @index(Group, Linear); li = @index(Local, Linear)
    i, j = _ij_from_li(li, Val(N))
    _face_sat_apply_2d!(Val(3), üe_loc, face_buf, i, j, ops, Val(N))
    @synchronize

    e    = @index(Group, Linear); li = @index(Local, Linear)
    i, j = _ij_from_li(li, Val(N))
    _face_sat_compute_2d!(Val(4), face_buf, i, j, e, geom, work, geom.conn, ops, τ, bdry_values, H_1d, Val(N))
    @synchronize
    e    = @index(Group, Linear); li = @index(Local, Linear)
    i, j = _ij_from_li(li, Val(N))
    _face_sat_apply_2d!(Val(4), üe_loc, face_buf, i, j, ops, Val(N))
    @synchronize

    # ---- Mass division + global writeback ----
    e    = @index(Group, Linear); li = @index(Local, Linear)
    i, j = _ij_from_li(li, Val(N))
    @inbounds üe_loc[i, j] /= geom.Hphys[i, j, e]
    @inbounds ü[i, j, e] = üe_loc[i, j]
end

function apply_laplacian!(ü::AbstractArray{T,3}, u::AbstractArray{T,3},
                             bdry_values::NTuple{4, T};
                             geom::MeshGeometry{2, T, N},
                             ops::SBPOps{N, T},
                             work::MeshWorkspace{2, T, N},
                             τ) where {N, T}
    @assert size(ü) == size(u)
    @assert size(u, 1) == size(u, 2) == N
    @assert size(u, 3) == geom.Ne

    H_1d = SVector{N, T}(ntuple(i -> ops.H[i, i], Val(N)))
    backend = get_backend(u)

    _laplacian2d_face_trace_kernel!(backend, N^2)(
        u, geom, work, ops, Val(N);
        ndrange = N^2 * geom.Ne)
    _laplacian2d_volume_kernel!(backend, N^2)(
        ü, u, geom, work, ops, T(τ), bdry_values, H_1d, Val(N);
        ndrange = N^2 * geom.Ne)

    return ü
end

################################################################################
# Diagnostics (2D)

"""
    discrete_laplacian(geom, ops, τ;
                         bdry_values = ntuple(_ -> 0, 4),
                         drop_tol    = 0) → SparseMatrixCSC

2D analog of [`discrete_laplacian`](@ref). Assembles `L_h` as an
explicit sparse matrix by running `apply_laplacian!` on each
canonical basis vector. Host-only.
"""
function discrete_laplacian(geom::MeshGeometry{2, T, N}, ops::SBPOps{N, T}, τ;
                              bdry_values::NTuple{4, T} = ntuple(_ -> zero(T), Val(4)),
                              drop_tol = zero(T)) where {N, T}
    Ne   = geom.Ne
    ndof = N^2 * Ne
    u  = zeros(T, N, N, Ne)
    ü = similar(u)
    work = make_workspace(geom)

    I_idx = Int[];  J_idx = Int[];  V = T[]
    sizehint!(I_idx, 32 * ndof);  sizehint!(J_idx, 32 * ndof);  sizehint!(V, 32 * ndof)

    u_flat = vec(u)
    ü_flat = vec(ü)
    @inbounds for col in 1:ndof
        u_flat[col] = one(T)
        apply_laplacian!(ü, u, bdry_values; geom, ops, work, τ)
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
                               tol, maxiter = 50, krylovdim = 30) → T

2D analog of [`spectral_radius_estimate`](@ref). Estimates `|λ_max|`
of the 2D discrete Laplacian via `KrylovKit.eigsolve(:LM)` on a
matrix-free shell wrapping `apply_laplacian!`.

`T` is restricted to `Float32` / `Float64` for the same KrylovKit-LAPACK
reason documented on the 3D version.
"""
function spectral_radius_estimate(geom::MeshGeometry{2, T, N}, ops::SBPOps{N, T}, τ;
                                    tol = eigsolve_default_tol(T),
                                    maxiter::Int = 50,
                                    krylovdim::Int = 30) where {N, T}
    Ne   = geom.Ne
    bdry = ntuple(_ -> zero(T), Val(4))

    backend = get_backend(geom.coords)
    work    = make_workspace(geom)

    x₀ = KernelAbstractions.allocate(backend, T, N, N, Ne)
    Random.randn!(x₀)
    n0 = sqrt(sum(abs2, x₀))
    n0 == 0 && return zero(T)
    x₀ ./= n0

    apply_Lh = let geom = geom, ops = ops, work = work, τ = τ, bdry = bdry
        x -> begin
            y = similar(x)
            apply_laplacian!(y, x, bdry; geom, ops, work, τ)
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

2D overload of [`discrete_inner_product`](@ref). `u`, `v` are 3-D
arrays of shape `(N, N, Ne)`.
"""
function discrete_inner_product(u::AbstractArray{T, 3}, v::AbstractArray{T, 3},
                                 geom::MeshGeometry{2, T, N},
                                 ops::SBPOps{N, T}) where {N, T}
    @assert size(u) == size(v) == size(geom.Hphys) == (N, N, geom.Ne)
    return mapreduce((uᵢ, vᵢ, hᵢ) -> uᵢ * vᵢ * hᵢ, +, u, v, geom.Hphys;
                     init = zero(T))
end

"""
    discrete_l2_norm(u, geom, ops) → T

2D overload. See [`discrete_inner_product`](@ref).
"""
discrete_l2_norm(u::AbstractArray{T, 3}, geom::MeshGeometry{2, T, N},
                  ops::SBPOps{N, T}) where {N, T} =
    sqrt(discrete_inner_product(u, u, geom, ops))
