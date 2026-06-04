# 2D first-derivative operator: physical d-derivative (d ∈ {1, 2})
# with centred-flux SAT, the building block for the conservative
# first-order scalar-wave gradient/divergence pair. Mirrors the 1D
# `apply_D!`: reference SBP-G along the chosen reference axis +
# centred-flux SAT at the two faces normal to that axis, neighbour
# relation from `geom.conn`.
#
# On axis-aligned **affine** meshes (`make_uniform_quad`) the inverse
# Jacobian is diagonal and constant per element, so the d-derivative
# decouples from the transverse axis and is exactly the 1D skew
# operator applied along axis d (`D_1d ⊗ I`); the assembled `H·D` is
# therefore exactly skew (skew ⊗ symmetric). The SAT coefficient is
# `½·invjac[d,d]/H_1d[face]` — the 1D-direction weight, NOT the 2D
# node mass `Hphys` (which carries the extra transverse factor).
#
# Curvilinear meshes (full, node-varying invjac) are NOT handled here:
# a conservative skew first derivative there needs discrete metric
# identities / free-stream preservation. Deferred.

# Faces normal to reference axis d: (−d face, +d face).
@inline _axis_faces(d) = (2d - 1, 2d)

# Along-axis node index of a neighbour at its touching face `nf`
# (−faces 1,3,5 → node 1; +faces 2,4,6 → node N).
@inline _face_node(nf, ::Val{N}) where {N} = isodd(nf) ? 1 : N

# ---- Shared face-node gather (pass 1 of the two-pass 2D operators) ----
# Write each element's INTERIOR-face node values into the workspace face
# trace, following the `_facenode2d` convention (face 1,2 → p = η index j;
# face 3,4 → p = ξ index i). Outer faces (`bdry ≠ 0`) carry no SAT and are
# skipped. Used by `apply_D!` (1 channel) and `apply_gradient2d!` /
# `apply_divergence2d!` (1 / 2 channels) — the single source of the
# inter-element communication.
@kernel function _gather_face2d_1ch!(@Const(u), work, @Const(bdry),
                                     ::Val{N}) where {N}
    i, j, e = @index(Global, NTuple)
    @inbounds begin
        v = u[i, j, e]
        if i == 1 && bdry[1, e] == 0; work.face_trace[1, j, 1, e] = v; end
        if i == N && bdry[2, e] == 0; work.face_trace[1, j, 2, e] = v; end
        if j == 1 && bdry[3, e] == 0; work.face_trace[1, i, 3, e] = v; end
        if j == N && bdry[4, e] == 0; work.face_trace[1, i, 4, e] = v; end
    end
end

@kernel function _gather_face2d_2ch!(@Const(F1), @Const(F2), work,
                                     @Const(bdry), ::Val{N}) where {N}
    i, j, e = @index(Global, NTuple)
    @inbounds begin
        v1 = F1[i, j, e]; v2 = F2[i, j, e]
        if i == 1 && bdry[1, e] == 0
            work.face_trace[1, j, 1, e] = v1; work.face_trace[2, j, 1, e] = v2
        end
        if i == N && bdry[2, e] == 0
            work.face_trace[1, j, 2, e] = v1; work.face_trace[2, j, 2, e] = v2
        end
        if j == 1 && bdry[3, e] == 0
            work.face_trace[1, i, 3, e] = v1; work.face_trace[2, i, 3, e] = v2
        end
        if j == N && bdry[4, e] == 0
            work.face_trace[1, i, 4, e] = v1; work.face_trace[2, i, 4, e] = v2
        end
    end
end

"""
    apply_D!(Du, u, d::Integer; geom::MeshGeometry{2, T, N}, ops, work) → Du

Physical `d`-th partial derivative (`d ∈ {1, 2}`) of the scalar field
`u :: (N, N, Ne)`, written into `Du`, via reference SBP-G along axis
`d` plus centred-flux SAT at the two faces normal to `d` (neighbour
relation from `geom.conn`). Axis-aligned affine meshes only (diagonal
`invjac`); `H·D` is then exactly skew. Two KernelAbstractions passes
(gather → volume+SAT) that run on both CPU and GPU; `work` supplies the
face-trace buffer (channel 1).
"""
function apply_D!(Du::AbstractArray{T,3}, u::AbstractArray{T,3}, d::Integer;
                  geom::MeshGeometry{2, T, N}, ops::SBPOps{N, T},
                  work::MeshWorkspace{2, T, N}) where {N, T}
    @assert size(u) == size(Du) == (N, N, geom.Ne)
    @assert d == 1 || d == 2
    backend = get_backend(u)
    _gather_face2d_1ch!(backend, (N, N))(
        u, work, geom.conn.bdry, Val(N); ndrange = (N, N, geom.Ne))
    _apply_D_2d_volume_kernel!(backend, N^2)(
        Du, u, work, ops, geom.conn.neighbour, geom.conn.neighbour_face,
        geom.conn.orientation, geom.conn.bdry, geom.invjac,
        Val(Int(d)), Val(N); ndrange = N^2 * geom.Ne)
    return Du
end

# Pass 2: per element, stage u into `@localmem`, reference d-derivative ×
# invjac[d,d], then centred-flux SAT at the two faces normal to d, reading
# the neighbour's gathered trace. One write per node.
@kernel function _apply_D_2d_volume_kernel!(Du::AbstractArray{T}, @Const(u),
                                            work, ops, @Const(neighbour),
                                            @Const(nbr_face), @Const(orient),
                                            @Const(bdry), @Const(invjac),
                                            ::Val{d}, ::Val{N}) where {T, d, N}
    e    = @index(Group, Linear)
    li   = @index(Local, Linear)
    i, j = _ij_from_li(li, Val(N))
    u_loc = @localmem T (N, N)
    @inbounds u_loc[i, j] = u[i, j, e]
    @synchronize
    e    = @index(Group, Linear)
    li   = @index(Local, Linear)
    i, j = _ij_from_li(li, Val(N))
    half = T(1) / T(2)
    G = ops.G; H1 = ops.H

    # Volume term.
    s = zero(T)
    if d == 1
        @inbounds for p in 1:N
            s += G[i, p] * u_loc[p, j]
        end
        @inbounds s *= invjac[1, 1, i, j, e]
    else
        @inbounds for p in 1:N
            s += G[j, p] * u_loc[i, p]
        end
        @inbounds s *= invjac[2, 2, i, j, e]
    end

    # Centred-flux SAT at the two faces normal to axis d.
    @inbounds begin
        on_lo = (d == 1) ? (i == 1) : (j == 1)
        on_hi = (d == 1) ? (i == N) : (j == N)
        if on_lo && bdry[2d - 1, e] == 0
            nbr = Int(neighbour[2d-1, e]); nf = Int(nbr_face[2d-1, e])
            o   = Int(orient[2d-1, e])
            t   = (d == 1) ? j : i
            tn  = _neigh_p(o, t, N)
            u_nbr = work.face_trace[1, tn, nf, nbr]
            s += (u_loc[i, j] - u_nbr) * half * invjac[d, d, i, j, e] / H1[1, 1]
        end
        if on_hi && bdry[2d, e] == 0
            nbr = Int(neighbour[2d, e]); nf = Int(nbr_face[2d, e])
            o   = Int(orient[2d, e])
            t   = (d == 1) ? j : i
            tn  = _neigh_p(o, t, N)
            u_nbr = work.face_trace[1, tn, nf, nbr]
            s += (u_nbr - u_loc[i, j]) * half * invjac[d, d, i, j, e] / H1[N, N]
        end
    end

    @inbounds Du[i, j, e] = s
end
