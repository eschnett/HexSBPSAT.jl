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

"""
    apply_D!(Du, u, d::Integer; geom::MeshGeometry{2, T, N}, ops) → Du

Physical `d`-th partial derivative (`d ∈ {1, 2}`) of the scalar field
`u :: (N, N, Ne)`, written into `Du`, via reference SBP-G along axis
`d` plus centred-flux SAT at the two faces normal to `d` (neighbour
relation from `geom.conn`). Axis-aligned affine meshes only (diagonal
`invjac`); `H·D` is then exactly skew.
"""
function apply_D!(Du::AbstractArray{T,3}, u::AbstractArray{T,3}, d::Integer;
                  geom::MeshGeometry{2, T, N}, ops::SBPOps{N, T}) where {N, T}
    @assert size(u) == size(Du) == (N, N, geom.Ne)
    @assert d == 1 || d == 2
    backend = get_backend(u)
    if backend isa KernelAbstractions.CPU
        return _apply_D_2d_cpu!(Du, u, d, geom, ops)
    end
    _apply_D_2d_kernel!(backend, (N, N))(
        Du, u, ops, geom.conn.neighbour, geom.conn.neighbour_face,
        geom.conn.orientation, geom.conn.bdry, geom.invjac,
        Val(Int(d)), Val(N); ndrange = (N, N, geom.Ne))
    return Du
end

# KA kernel — workgroup-per-element, N² workitems; mirrors the CPU
# per-element flow. `u_loc` stages the element into shared memory for
# the volume stencil; neighbour face values are read from global `u`.
@kernel function _apply_D_2d_kernel!(Du::AbstractArray{T,3},
                                     @Const(u::AbstractArray{T,3}),
                                     ops, @Const(neighbour),
                                     @Const(nbr_face), @Const(orient),
                                     @Const(bdry), @Const(invjac),
                                     ::Val{d}, ::Val{N}) where {T, d, N}
    i, j, m = @index(Global, NTuple)
    il, jl = @index(Local, NTuple)

    u_loc = @localmem T (N, N)
    @inbounds u_loc[il, jl] = u[i, j, m]
    @synchronize

    i, j, m = @index(Global, NTuple)
    il, jl = @index(Local, NTuple)
    half = T(1) / T(2)
    G = ops.G; H1 = ops.H

    # Volume term.
    s = zero(T)
    if d == 1
        @inbounds for p in 1:N
            s += G[il, p] * u_loc[p, jl]
        end
        @inbounds s *= invjac[1, 1, i, j, m]
    else
        @inbounds for p in 1:N
            s += G[jl, p] * u_loc[il, p]
        end
        @inbounds s *= invjac[2, 2, i, j, m]
    end

    # Centred-flux SAT at the two faces normal to axis d.
    fm = 2d - 1; fp = 2d
    @inbounds begin
        on_lo = (d == 1) ? (i == 1) : (j == 1)
        on_hi = (d == 1) ? (i == N) : (j == N)
        if on_lo && bdry[fm, m] == 0
            nbr = Int(neighbour[fm, m]); nf = Int(nbr_face[fm, m])
            o   = Int(orient[fm, m]);    nn = isodd(nf) ? 1 : N
            t   = (d == 1) ? j : i
            tn  = _neigh_p(o, t, N)
            u_self = u_loc[il, jl]
            u_nbr  = (d == 1) ? u[nn, tn, nbr] : u[tn, nn, nbr]
            s += (u_self - u_nbr) * half * invjac[d, d, i, j, m] / H1[1, 1]
        end
        if on_hi && bdry[fp, m] == 0
            nbr = Int(neighbour[fp, m]); nf = Int(nbr_face[fp, m])
            o   = Int(orient[fp, m]);    nn = isodd(nf) ? 1 : N
            t   = (d == 1) ? j : i
            tn  = _neigh_p(o, t, N)
            u_self = u_loc[il, jl]
            u_nbr  = (d == 1) ? u[nn, tn, nbr] : u[tn, nn, nbr]
            s += (u_nbr - u_self) * half * invjac[d, d, i, j, m] / H1[N, N]
        end
    end

    @inbounds Du[i, j, m] = s
end

@inline function _apply_D_2d_cpu!(Du::AbstractArray{T,3}, u::AbstractArray{T,3},
                                  d::Integer, geom::MeshGeometry{2, T, N},
                                  ops::SBPOps{N, T}) where {N, T}
    Ne        = geom.Ne
    neighbour = geom.conn.neighbour
    nbr_face  = geom.conn.neighbour_face
    orient    = geom.conn.orientation
    bdry      = geom.conn.bdry
    half      = one(T) / 2
    fm, fp    = _axis_faces(d)            # −d face, +d face
    G         = ops.G
    H1        = ops.H

    @inbounds for m in 1:Ne
        # Volume term: reference d-derivative × invjac[d,d] (diagonal).
        if d == 1
            for j in 1:N, i in 1:N
                s = zero(T)
                for p in 1:N
                    s += G[i, p] * u[p, j, m]
                end
                Du[i, j, m] = s * geom.invjac[1, 1, i, j, m]
            end
        else
            for j in 1:N, i in 1:N
                s = zero(T)
                for p in 1:N
                    s += G[j, p] * u[i, p, m]
                end
                Du[i, j, m] = s * geom.invjac[2, 2, i, j, m]
            end
        end

        # Centred-flux SAT at the −d and +d faces. Coefficient uses the
        # 1D-direction weight H_1d[face] and the along-axis invjac.
        # −d face (along-axis node 1):
        if bdry[fm, m] == 0
            nbr = Int(neighbour[fm, m])
            nf  = Int(nbr_face[fm, m])
            o   = Int(orient[fm, m])
            nn  = _face_node(nf, Val(N))
            for t in 1:N                      # transverse node on self
                tn = _neigh_p(o, t, N)        # transverse node on neighbour
                if d == 1
                    u_self = u[1, t, m]
                    u_nbr  = u[nn, tn, nbr]
                    Du[1, t, m] += (u_self - u_nbr) * half *
                                   geom.invjac[1, 1, 1, t, m] / H1[1, 1]
                else
                    u_self = u[t, 1, m]
                    u_nbr  = u[tn, nn, nbr]
                    Du[t, 1, m] += (u_self - u_nbr) * half *
                                   geom.invjac[2, 2, t, 1, m] / H1[1, 1]
                end
            end
        end
        # +d face (along-axis node N):
        if bdry[fp, m] == 0
            nbr = Int(neighbour[fp, m])
            nf  = Int(nbr_face[fp, m])
            o   = Int(orient[fp, m])
            nn  = _face_node(nf, Val(N))
            for t in 1:N
                tn = _neigh_p(o, t, N)
                if d == 1
                    u_self = u[N, t, m]
                    u_nbr  = u[nn, tn, nbr]
                    Du[N, t, m] += (u_nbr - u_self) * half *
                                   geom.invjac[1, 1, N, t, m] / H1[N, N]
                else
                    u_self = u[t, N, m]
                    u_nbr  = u[tn, nn, nbr]
                    Du[t, N, m] += (u_nbr - u_self) * half *
                                   geom.invjac[2, 2, t, N, m] / H1[N, N]
                end
            end
        end
    end
    return Du
end
