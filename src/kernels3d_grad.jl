# 3D first-derivative operator: physical d-derivative (d ∈ {1,2,3}) with
# centred-flux SAT — the 3D analog of the 2D affine `apply_D!`
# (kernels2d_grad.jl), the building block for the conservative
# first-order scalar-wave gradient/divergence on axis-aligned (affine)
# hex meshes.
#
# On axis-aligned affine meshes (`make_uniform_hex`) the inverse Jacobian
# is diagonal and constant per element, so the d-derivative is exactly
# the 1D skew operator applied along reference axis d; the assembled
# `H·D` is exactly skew. SAT coefficient `½·invjac[d,d]/H_1d[face]` — the
# 1D-direction weight, not the 3D node mass.
#
# Curvilinear meshes (node-varying invjac) need discrete metric identities
# in the conservative-curl form for free-stream preservation — handled by
# the separate `apply_gradient3d!`/`apply_divergence3d!` (future work),
# not here.
#
# Single KA two-pass kernel set (gather → volume+SAT) run on both CPU and
# GPU, workgroup-per-element with N³ workitems, mirroring the 3D Laplacian
# (`_ijk_from_li`, `_neigh_pq` D₄ orientation, `work.face_trace`).

# ---- Shared face-node gather (pass 1) ----
# Write each element's interior-face node values into the workspace face
# trace `work.face_trace[1, p, q, f, e]`, following the Laplacian's
# tangential convention (face normal to axis 1 → (p,q)=(j,k); axis 2 →
# (i,k); axis 3 → (i,j)). Outer faces (`bdry ≠ 0`) carry no SAT and are
# skipped.
@kernel function _gather_face3d_1ch!(@Const(u), work, @Const(bdry),
                                     ::Val{N}) where {N}
    i, j, k, e = @index(Global, NTuple)
    @inbounds begin
        v = u[i, j, k, e]
        if i == 1 && bdry[1, e] == 0; work.face_trace[1, j, k, 1, e] = v; end
        if i == N && bdry[2, e] == 0; work.face_trace[1, j, k, 2, e] = v; end
        if j == 1 && bdry[3, e] == 0; work.face_trace[1, i, k, 3, e] = v; end
        if j == N && bdry[4, e] == 0; work.face_trace[1, i, k, 4, e] = v; end
        if k == 1 && bdry[5, e] == 0; work.face_trace[1, i, j, 5, e] = v; end
        if k == N && bdry[6, e] == 0; work.face_trace[1, i, j, 6, e] = v; end
    end
end

"""
    apply_D!(Du, u, d::Integer; geom::MeshGeometry{3, T, N}, ops, work) → Du

Physical `d`-th partial derivative (`d ∈ {1, 2, 3}`) of the scalar field
`u :: (N, N, N, Ne)`, written into `Du`, via reference SBP-G along axis
`d` plus centred-flux SAT at the two faces normal to `d`. Axis-aligned
affine meshes only (diagonal `invjac`); `H·D` is then exactly skew. Two
KernelAbstractions passes (gather → volume+SAT) that run on both CPU and
GPU; `work` supplies the face-trace buffer (channel 1).
"""
function apply_D!(Du::AbstractArray{T,4}, u::AbstractArray{T,4}, d::Integer;
                  geom::MeshGeometry{3, T, N}, ops::SBPOps{N, T},
                  work::MeshWorkspace{3, T, N}) where {N, T}
    @assert size(u) == size(Du) == (N, N, N, geom.Ne)
    @assert 1 ≤ d ≤ 3
    backend = get_backend(u)
    _gather_face3d_1ch!(backend, (N, N, N))(
        u, work, geom.conn.bdry, Val(N); ndrange = (N, N, N, geom.Ne))
    _apply_D_3d_volume_kernel!(backend, N^3)(
        Du, u, work, ops, geom.conn.neighbour, geom.conn.neighbour_face,
        geom.conn.orientation, geom.conn.bdry, geom.invjac,
        Val(Int(d)), Val(N); ndrange = N^3 * geom.Ne)
    return Du
end

# Pass 2: per element, stage u into `@localmem`, reference d-derivative ×
# invjac[d,d], then centred-flux SAT at the two faces normal to d, reading
# the neighbour's gathered trace (orientation via `_neigh_pq`). One write
# per node.
@kernel function _apply_D_3d_volume_kernel!(Du::AbstractArray{T}, @Const(u),
                                            work, ops, @Const(neighbour),
                                            @Const(nbr_face), @Const(orient),
                                            @Const(bdry), @Const(invjac),
                                            ::Val{d}, ::Val{N}) where {T, d, N}
    e       = @index(Group, Linear)
    li      = @index(Local, Linear)
    i, j, k = _ijk_from_li(li, Val(N))
    u_loc = @localmem T (N, N, N)
    @inbounds u_loc[i, j, k] = u[i, j, k, e]
    @synchronize
    e       = @index(Group, Linear)
    li      = @index(Local, Linear)
    i, j, k = _ijk_from_li(li, Val(N))
    half = T(1) / T(2)
    G = ops.G; H1 = ops.H

    # Volume term: reference d-derivative × invjac[d,d] (diagonal/affine).
    s = zero(T)
    if d == 1
        @inbounds for p in 1:N
            s += G[i, p] * u_loc[p, j, k]
        end
        @inbounds s *= invjac[1, 1, i, j, k, e]
    elseif d == 2
        @inbounds for p in 1:N
            s += G[j, p] * u_loc[i, p, k]
        end
        @inbounds s *= invjac[2, 2, i, j, k, e]
    else
        @inbounds for p in 1:N
            s += G[k, p] * u_loc[i, j, p]
        end
        @inbounds s *= invjac[3, 3, i, j, k, e]
    end

    # Centred-flux SAT at the two faces normal to axis d.
    @inbounds begin
        ia    = d == 1 ? i : d == 2 ? j : k
        idJ   = invjac[d, d, i, j, k, e]
        if ia == 1 && bdry[2d - 1, e] == 0
            nbr = Int(neighbour[2d-1, e]); nf = Int(nbr_face[2d-1, e])
            o   = orient[2d-1, e]
            p, q = d == 1 ? (j, k) : d == 2 ? (i, k) : (i, j)
            pn, qn = _neigh_pq(o, p, q, Int32(N))
            u_nbr = work.face_trace[1, pn, qn, nf, nbr]
            s += (u_loc[i, j, k] - u_nbr) * half * idJ / H1[1, 1]
        end
        if ia == N && bdry[2d, e] == 0
            nbr = Int(neighbour[2d, e]); nf = Int(nbr_face[2d, e])
            o   = orient[2d, e]
            p, q = d == 1 ? (j, k) : d == 2 ? (i, k) : (i, j)
            pn, qn = _neigh_pq(o, p, q, Int32(N))
            u_nbr = work.face_trace[1, pn, qn, nf, nbr]
            s += (u_nbr - u_loc[i, j, k]) * half * idJ / H1[N, N]
        end
    end

    @inbounds Du[i, j, k, e] = s
end
