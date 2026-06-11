# Channel-batched variants of the 3D first-derivative operators: the
# same two-pass (face-trace gather → volume+SAT) kernels as
# kernels3d_grad.jl / kernels3d_curv.jl, but operating on C fields at
# once with a trailing channel dimension, `u :: (N, N, N, Ne, C)`.
#
# Motivation: multi-field systems (e.g. the 10/20-component generalized
# harmonic state) otherwise pay one kernel launch per component per
# operator. On the KernelAbstractions CPU backend every launch costs a
# task-spawn + barrier across all threads, so collapsing 10–20 launches
# into one matters at high thread counts; on GPUs it amortises launch
# latency. The volume kernels use one workgroup per (element, channel)
# pair — same N³ workitems and identical local-memory footprint as the
# scalar kernels — and the math matches the scalar kernels exactly
# (asserted to round-off in test/test_kernels3d_batch.jl).
#
# The face-trace workspace needs `nchannels ≥ C` for gather/gradient and
# `nchannels ≥ 3C` for the divergence; allocate via
# `make_workspace(geom; nchannels)`.


# The batched kernels index the face-trace pool with the channel as the
# LAST dimension, so both the field reads and the trace writes stream
# contiguously (the scalar kernels keep the channels-first convention).
# Same memory, different index map.
@inline function _trace_channels_last(work::MeshWorkspace{3, T, N},
                                      Ne::Int, ::Val{N}) where {T, N}
    nch = size(work.face_trace, 1)
    return reshape(work.face_trace, N, N, 6, Ne, nch)
end

# ---- Pass 1: batched face gathers -------------------------------------

# Gather all C channels of u into trace channels 1..C. 5-D ndrange so
# consecutive workitems stream contiguously through u.
@kernel function _gather_face3d_batch!(@Const(u), ft, @Const(bdry),
                                       ::Val{N}) where {N}
    i, j, k, e, c = @index(Global, NTuple)
    @inbounds begin
        v = u[i, j, k, e, c]
        if i == 1 && bdry[1, e] == 0; ft[j, k, 1, e, c] = v; end
        if i == N && bdry[2, e] == 0; ft[j, k, 2, e, c] = v; end
        if j == 1 && bdry[3, e] == 0; ft[i, k, 3, e, c] = v; end
        if j == N && bdry[4, e] == 0; ft[i, k, 4, e, c] = v; end
        if k == 1 && bdry[5, e] == 0; ft[i, j, 5, e, c] = v; end
        if k == N && bdry[6, e] == 0; ft[i, j, 6, e, c] = v; end
    end
end

# Gather the three flux fields into trace channels c, C+c, 2C+c.
@kernel function _gather_face3d_batch3!(@Const(F1), @Const(F2), @Const(F3),
                                        ft, @Const(bdry),
                                        ::Val{N}) where {N}
    i, j, k, e, c = @index(Global, NTuple)
    C = size(F1, 5)
    @inbounds begin
        v1 = F1[i, j, k, e, c]; v2 = F2[i, j, k, e, c]; v3 = F3[i, j, k, e, c]
        if i == 1 && bdry[1, e] == 0
            ft[j, k, 1, e, c] = v1; ft[j, k, 1, e, C+c] = v2; ft[j, k, 1, e, 2C+c] = v3
        end
        if i == N && bdry[2, e] == 0
            ft[j, k, 2, e, c] = v1; ft[j, k, 2, e, C+c] = v2; ft[j, k, 2, e, 2C+c] = v3
        end
        if j == 1 && bdry[3, e] == 0
            ft[i, k, 3, e, c] = v1; ft[i, k, 3, e, C+c] = v2; ft[i, k, 3, e, 2C+c] = v3
        end
        if j == N && bdry[4, e] == 0
            ft[i, k, 4, e, c] = v1; ft[i, k, 4, e, C+c] = v2; ft[i, k, 4, e, 2C+c] = v3
        end
        if k == 1 && bdry[5, e] == 0
            ft[i, j, 5, e, c] = v1; ft[i, j, 5, e, C+c] = v2; ft[i, j, 5, e, 2C+c] = v3
        end
        if k == N && bdry[6, e] == 0
            ft[i, j, 6, e, c] = v1; ft[i, j, 6, e, C+c] = v2; ft[i, j, 6, e, 2C+c] = v3
        end
    end
end

# ---- Batched axis derivative (affine meshes) ---------------------------

"""
    apply_D_batch!(Du, u, d::Integer; geom::MeshGeometry{3, T, N}, ops,
                   work, scale = one(T), accumulate = Val(false)) → Du

Channel-batched form of [`apply_D!`](@ref): physical `d`-th partial
derivative of every channel of `u :: (N, N, N, Ne, C)`, written into
`Du` (same shape) as

    Du[…, c] = scale · D_d u[…, c]            (accumulate = Val(false))
    Du[…, c] += scale · D_d u[…, c]           (accumulate = Val(true))

Axis-aligned affine meshes only (diagonal `invjac`), exactly the scalar
operator per channel. One gather + one volume launch regardless of C;
`work` must have `nchannels ≥ C`. The fused `scale`/`accumulate` cover
divergence assembly and dissipation accumulation without intermediate
broadcasts.
"""
function apply_D_batch!(Du::AbstractArray{T,5}, u::AbstractArray{T,5},
                        d::Integer; geom::MeshGeometry{3, T, N},
                        ops::SBPOps{N, T}, work::MeshWorkspace{3, T, N},
                        scale = one(T),
                        accumulate::Val = Val(false),
                        backend = get_backend(u)) where {N, T}
    C = size(u, 5)
    @assert size(u) == size(Du) == (N, N, N, geom.Ne, C)
    @assert 1 <= d <= 3
    # CPU fast path (kernels3d_batch_cpu.jl): SIMD pencil loops, direct
    # neighbour reads (the face-trace pool is NOT populated), bitwise-
    # identical results. GPU backends use the KA kernels.
    # HEXSBPSAT_NO_CPU_FASTPATH=1 forces the KA path (benchmarking only).
    if backend isa CPU && get(ENV, "HEXSBPSAT_NO_CPU_FASTPATH", "0") != "1"
        return _apply_D_batch_cpu!(Du, u, Val(Int(d)); geom, ops,
                                   scale = T(scale), accumulate)
    end
    return _apply_D_batch_ka!(Du, u, d; geom, ops, work,
                              scale = T(scale), accumulate, backend)
end

# The KernelAbstractions implementation — the GPU path, also directly
# callable on CPU (the bitwise KA-vs-fast-path tests rely on this).
function _apply_D_batch_ka!(Du::AbstractArray{T,5}, u::AbstractArray{T,5},
                            d::Integer; geom::MeshGeometry{3, T, N},
                            ops::SBPOps{N, T}, work::MeshWorkspace{3, T, N},
                            scale::T = one(T),
                            accumulate::Val = Val(false),
                            backend = get_backend(u)) where {N, T}
    C = size(u, 5)
    @assert size(work.face_trace, 1) >= C
    ft = _trace_channels_last(work, geom.Ne, Val(N))
    _gather_face3d_batch!(backend, (N, N, N))(
        u, ft, geom.conn.bdry, Val(N); ndrange = (N, N, N, geom.Ne, C))
    _apply_D_3d_volume_batch_kernel!(backend, (N^3, 1))(
        Du, u, ft, ops, geom.conn.neighbour, geom.conn.neighbour_face,
        geom.conn.orientation, geom.conn.bdry, geom.invjacd, scale,
        accumulate, Val(Int(d)), Val(N); ndrange = (N^3 * geom.Ne, C))
    return Du
end

@kernel function _apply_D_3d_volume_batch_kernel!(
        Du::AbstractArray{T}, @Const(u), ft, ops, @Const(neighbour),
        @Const(nbr_face), @Const(orient), @Const(bdry), @Const(invjacd),
        scale::T, ::Val{ACC}, ::Val{d}, ::Val{N}) where {T, ACC, d, N}
    ge = @index(Group, NTuple)
    li = @index(Local, Linear)
    e = ge[1]; c = ge[2]
    i, j, k = _ijk_from_li(li, Val(N))
    u_loc = @localmem T (N, N, N)
    @inbounds u_loc[i, j, k] = u[i, j, k, e, c]
    @synchronize
    ge = @index(Group, NTuple)
    li = @index(Local, Linear)
    e = ge[1]; c = ge[2]
    i, j, k = _ijk_from_li(li, Val(N))
    half = T(1) / T(2)
    G = ops.G; H1 = ops.H

    s = zero(T)
    if d == 1
        @inbounds for p in 1:N
            s += G[i, p] * u_loc[p, j, k]
        end
        @inbounds s *= invjacd[i, j, k, e, 1]
    elseif d == 2
        @inbounds for p in 1:N
            s += G[j, p] * u_loc[i, p, k]
        end
        @inbounds s *= invjacd[i, j, k, e, 2]
    else
        @inbounds for p in 1:N
            s += G[k, p] * u_loc[i, j, p]
        end
        @inbounds s *= invjacd[i, j, k, e, 3]
    end

    @inbounds begin
        ia  = d == 1 ? i : d == 2 ? j : k
        idJ = invjacd[i, j, k, e, d]
        if ia == 1 && bdry[2d - 1, e] == 0
            nbr = Int(neighbour[2d-1, e]); nf = Int(nbr_face[2d-1, e])
            o   = orient[2d-1, e]
            p, q = d == 1 ? (j, k) : d == 2 ? (i, k) : (i, j)
            pn, qn = _neigh_pq(o, p, q, Int32(N))
            u_nbr = ft[pn, qn, nf, nbr, c]
            s += (u_loc[i, j, k] - u_nbr) * half * idJ / H1[1, 1]
        end
        if ia == N && bdry[2d, e] == 0
            nbr = Int(neighbour[2d, e]); nf = Int(nbr_face[2d, e])
            o   = orient[2d, e]
            p, q = d == 1 ? (j, k) : d == 2 ? (i, k) : (i, j)
            pn, qn = _neigh_pq(o, p, q, Int32(N))
            u_nbr = ft[pn, qn, nf, nbr, c]
            s += (u_nbr - u_loc[i, j, k]) * half * idJ / H1[N, N]
        end
    end

    if ACC
        @inbounds Du[i, j, k, e, c] += scale * s
    else
        @inbounds Du[i, j, k, e, c] = scale * s
    end
end

# ---- Batched curvilinear gradient --------------------------------------

"""
    apply_gradient3d_batch!(g1, g2, g3, u; geom, ops, metric, work)
        → (g1, g2, g3)

Channel-batched form of [`apply_gradient3d!`](@ref): the physical
gradient of every channel of `u :: (N, N, N, Ne, C)` on a curvilinear
mesh (conservative free-stream-preserving split form + centred-flux
SAT). One gather + one volume launch; `work` needs `nchannels ≥ C`.
"""
function apply_gradient3d_batch!(g1::AbstractArray{T,5},
                                 g2::AbstractArray{T,5},
                                 g3::AbstractArray{T,5},
                                 u::AbstractArray{T,5};
                                 geom::MeshGeometry{3,T,N}, ops::SBPOps{N,T},
                                 metric, work::MeshWorkspace{3,T,N},
                                 backend = get_backend(u)) where {T,N}
    C = size(u, 5)
    @assert size(u) == size(g1) == size(g2) == size(g3) ==
            (N, N, N, geom.Ne, C)
    # CPU fast path (kernels3d_batch_cpu.jl); bitwise-identical, leaves
    # the face-trace pool untouched. GPU backends use the KA kernels.
    if backend isa CPU && get(ENV, "HEXSBPSAT_NO_CPU_FASTPATH", "0") != "1"
        return _apply_gradient3d_batch_cpu!(g1, g2, g3, u; geom, ops, metric)
    end
    return _apply_gradient3d_batch_ka!(g1, g2, g3, u; geom, ops, metric,
                                       work, backend)
end

# The KernelAbstractions implementation — the GPU path, also directly
# callable on CPU (the bitwise KA-vs-fast-path tests rely on this).
function _apply_gradient3d_batch_ka!(g1::AbstractArray{T,5},
                                     g2::AbstractArray{T,5},
                                     g3::AbstractArray{T,5},
                                     u::AbstractArray{T,5};
                                     geom::MeshGeometry{3,T,N},
                                     ops::SBPOps{N,T},
                                     metric, work::MeshWorkspace{3,T,N},
                                     backend = get_backend(u)) where {T,N}
    C = size(u, 5)
    @assert size(work.face_trace, 1) >= C
    ft = _trace_channels_last(work, geom.Ne, Val(N))
    _gather_face3d_batch!(backend, (N, N, N))(
        u, ft, geom.conn.bdry, Val(N); ndrange = (N, N, N, geom.Ne, C))
    _grad3d_volume_batch_kernel!(backend, (N^3, 1))(
        g1, g2, g3, u, ft, ops,
        metric.ax1, metric.ax2, metric.ax3, metric.ay1, metric.ay2, metric.ay3,
        metric.az1, metric.az2, metric.az3, metric.invdetJ,
        geom.conn.neighbour, geom.conn.neighbour_face, geom.conn.orientation,
        geom.conn.bdry, Val(N); ndrange = (N^3 * geom.Ne, C))
    return g1, g2, g3
end

@kernel function _grad3d_volume_batch_kernel!(g1::AbstractArray{T}, g2, g3,
        @Const(u), ft, ops, @Const(ax1), @Const(ax2), @Const(ax3),
        @Const(ay1), @Const(ay2), @Const(ay3), @Const(az1), @Const(az2),
        @Const(az3), @Const(invdetJ), @Const(neighbour), @Const(nbr_face),
        @Const(orient), @Const(bdry), ::Val{N}) where {T, N}
    ge = @index(Group, NTuple); li = @index(Local, Linear)
    e = ge[1]; c = ge[2]
    i, j, k = _ijk_from_li(li, Val(N))
    u_loc = @localmem T (N, N, N)
    @inbounds u_loc[i, j, k] = u[i, j, k, e, c]
    @synchronize
    ge = @index(Group, NTuple); li = @index(Local, Linear)
    e = ge[1]; c = ge[2]
    i, j, k = _ijk_from_li(li, Val(N))
    G = ops.G; half = T(1) / 2
    csx = zero(T); csy = zero(T); csz = zero(T)
    gξ = zero(T); gη = zero(T); gζ = zero(T)
    @inbounds for p in 1:N
        Gi = G[i, p]; Gj = G[j, p]; Gk = G[k, p]
        up = u_loc[p, j, k]; uq = u_loc[i, p, k]; ur = u_loc[i, j, p]
        csx += Gi*ax1[p,j,k,e]*up + Gj*ax2[i,p,k,e]*uq + Gk*ax3[i,j,p,e]*ur
        csy += Gi*ay1[p,j,k,e]*up + Gj*ay2[i,p,k,e]*uq + Gk*ay3[i,j,p,e]*ur
        csz += Gi*az1[p,j,k,e]*up + Gj*az2[i,p,k,e]*uq + Gk*az3[i,j,p,e]*ur
        gξ += Gi*up; gη += Gj*uq; gζ += Gk*ur
    end
    @inbounds idJ = invdetJ[i, j, k, e]
    @inbounds r1 = half*(csx + ax1[i,j,k,e]*gξ + ax2[i,j,k,e]*gη + ax3[i,j,k,e]*gζ)*idJ
    @inbounds r2 = half*(csy + ay1[i,j,k,e]*gξ + ay2[i,j,k,e]*gη + ay3[i,j,k,e]*gζ)*idJ
    @inbounds r3 = half*(csz + az1[i,j,k,e]*gξ + az2[i,j,k,e]*gη + az3[i,j,k,e]*gζ)*idJ
    @inbounds for f in 1:6
        bdry[f, e] == 0 || continue
        a_idx = (f + 1) ÷ 2; row = isodd(f) ? 1 : N
        on = a_idx == 1 ? (i == row) : a_idx == 2 ? (j == row) : (k == row)
        on || continue
        p_, q_ = a_idx == 1 ? (j, k) : a_idx == 2 ? (i, k) : (i, j)
        nbr = Int(neighbour[f, e]); nf = Int(nbr_face[f, e]); o = orient[f, e]
        s_f = isodd(f) ? -one(T) : one(T)
        pn, qn = _neigh_pq(o, p_, q_, Int32(N))
        jump = ft[pn, qn, nf, nbr, c] - u_loc[i, j, k]
        axc = a_idx == 1 ? ax1[i,j,k,e] : a_idx == 2 ? ax2[i,j,k,e] : ax3[i,j,k,e]
        ayc = a_idx == 1 ? ay1[i,j,k,e] : a_idx == 2 ? ay2[i,j,k,e] : ay3[i,j,k,e]
        azc = a_idx == 1 ? az1[i,j,k,e] : a_idx == 2 ? az2[i,j,k,e] : az3[i,j,k,e]
        cf = idJ * half * jump / ops.H[row, row]
        r1 += cf * s_f * axc; r2 += cf * s_f * ayc; r3 += cf * s_f * azc
    end
    @inbounds g1[i, j, k, e, c] = r1
    @inbounds g2[i, j, k, e, c] = r2
    @inbounds g3[i, j, k, e, c] = r3
end

# ---- Batched curvilinear divergence ------------------------------------

"""
    apply_divergence3d_batch!(divF, F1, F2, F3; geom, ops, metric, work,
                              add = nothing) → divF

Channel-batched form of [`apply_divergence3d!`](@ref): the physical
divergence of every channel of the flux `(F1, F2, F3)`, each
`(N, N, N, Ne, C)`. When `add` is an array of the same shape,
`divF[…, c] = ∇·F[…, c] + add[…, c]` is fused into the volume kernel
(no separate accumulation pass). One gather + one volume launch; `work`
needs `nchannels ≥ 3C`.
"""
function apply_divergence3d_batch!(divF::AbstractArray{T,5},
                                   F1::AbstractArray{T,5},
                                   F2::AbstractArray{T,5},
                                   F3::AbstractArray{T,5};
                                   geom::MeshGeometry{3,T,N}, ops::SBPOps{N,T},
                                   metric, work::MeshWorkspace{3,T,N},
                                   add::Union{Nothing,AbstractArray{T,5}} =
                                       nothing,
                                   backend = get_backend(divF)) where {T,N}
    C = size(F1, 5)
    @assert size(F1) == size(F2) == size(F3) == size(divF) ==
            (N, N, N, geom.Ne, C)
    # CPU fast path (kernels3d_batch_cpu.jl); bitwise-identical, leaves
    # the face-trace pool untouched. GPU backends use the KA kernels.
    if backend isa CPU && get(ENV, "HEXSBPSAT_NO_CPU_FASTPATH", "0") != "1"
        return _apply_divergence3d_batch_cpu!(divF, F1, F2, F3; geom, ops,
                                              metric, add)
    end
    return _apply_divergence3d_batch_ka!(divF, F1, F2, F3; geom, ops,
                                         metric, work, add, backend)
end

# The KernelAbstractions implementation — the GPU path, also directly
# callable on CPU (the bitwise KA-vs-fast-path tests rely on this).
function _apply_divergence3d_batch_ka!(divF::AbstractArray{T,5},
                                       F1::AbstractArray{T,5},
                                       F2::AbstractArray{T,5},
                                       F3::AbstractArray{T,5};
                                       geom::MeshGeometry{3,T,N},
                                       ops::SBPOps{N,T},
                                       metric, work::MeshWorkspace{3,T,N},
                                       add::Union{Nothing,AbstractArray{T,5}} =
                                           nothing,
                                       backend = get_backend(divF)) where {T,N}
    C = size(F1, 5)
    @assert size(work.face_trace, 1) >= 3C
    ft = _trace_channels_last(work, geom.Ne, Val(N))
    _gather_face3d_batch3!(backend, (N, N, N))(
        F1, F2, F3, ft, geom.conn.bdry, Val(N);
        ndrange = (N, N, N, geom.Ne, C))
    hasadd = add !== nothing
    addf = hasadd ? add : divF      # dummy, never read when hasadd == false
    _div3d_volume_batch_kernel!(backend, (N^3, 1))(
        divF, F1, F2, F3, addf, Val(hasadd), ft, ops,
        metric.ax1, metric.ax2, metric.ax3, metric.ay1, metric.ay2, metric.ay3,
        metric.az1, metric.az2, metric.az3, metric.invdetJ,
        geom.conn.neighbour, geom.conn.neighbour_face, geom.conn.orientation,
        geom.conn.bdry, Val(N); ndrange = (N^3 * geom.Ne, C))
    return divF
end

@kernel function _div3d_volume_batch_kernel!(divF::AbstractArray{T},
        @Const(F1), @Const(F2), @Const(F3), @Const(add), ::Val{HASADD},
        ft, ops, @Const(ax1), @Const(ax2), @Const(ax3), @Const(ay1),
        @Const(ay2), @Const(ay3), @Const(az1), @Const(az2), @Const(az3),
        @Const(invdetJ), @Const(neighbour), @Const(nbr_face), @Const(orient),
        @Const(bdry), ::Val{N}) where {T, HASADD, N}
    ge = @index(Group, NTuple); li = @index(Local, Linear)
    e = ge[1]; c = ge[2]
    i, j, k = _ijk_from_li(li, Val(N))
    F1l = @localmem T (N, N, N)
    F2l = @localmem T (N, N, N)
    F3l = @localmem T (N, N, N)
    @inbounds F1l[i, j, k] = F1[i, j, k, e, c]
    @inbounds F2l[i, j, k] = F2[i, j, k, e, c]
    @inbounds F3l[i, j, k] = F3[i, j, k, e, c]
    @synchronize
    ge = @index(Group, NTuple); li = @index(Local, Linear)
    C = size(F1, 5)
    e = ge[1]; c = ge[2]
    i, j, k = _ijk_from_li(li, Val(N))
    G = ops.G; half = T(1) / 2
    cs = zero(T)
    gξ1 = zero(T); gη1 = zero(T); gζ1 = zero(T)
    gξ2 = zero(T); gη2 = zero(T); gζ2 = zero(T)
    gξ3 = zero(T); gη3 = zero(T); gζ3 = zero(T)
    @inbounds for p in 1:N
        Gi = G[i, p]; Gj = G[j, p]; Gk = G[k, p]
        Fξ = ax1[p,j,k,e]*F1l[p,j,k] + ay1[p,j,k,e]*F2l[p,j,k] + az1[p,j,k,e]*F3l[p,j,k]
        Fη = ax2[i,p,k,e]*F1l[i,p,k] + ay2[i,p,k,e]*F2l[i,p,k] + az2[i,p,k,e]*F3l[i,p,k]
        Fζ = ax3[i,j,p,e]*F1l[i,j,p] + ay3[i,j,p,e]*F2l[i,j,p] + az3[i,j,p,e]*F3l[i,j,p]
        cs += Gi*Fξ + Gj*Fη + Gk*Fζ
        gξ1 += Gi*F1l[p,j,k]; gη1 += Gj*F1l[i,p,k]; gζ1 += Gk*F1l[i,j,p]
        gξ2 += Gi*F2l[p,j,k]; gη2 += Gj*F2l[i,p,k]; gζ2 += Gk*F2l[i,j,p]
        gξ3 += Gi*F3l[p,j,k]; gη3 += Gj*F3l[i,p,k]; gζ3 += Gk*F3l[i,j,p]
    end
    @inbounds idJ = invdetJ[i, j, k, e]
    @inbounds ad = ax1[i,j,k,e]*gξ1 + ax2[i,j,k,e]*gη1 + ax3[i,j,k,e]*gζ1 +
                   ay1[i,j,k,e]*gξ2 + ay2[i,j,k,e]*gη2 + ay3[i,j,k,e]*gζ2 +
                   az1[i,j,k,e]*gξ3 + az2[i,j,k,e]*gη3 + az3[i,j,k,e]*gζ3
    r = half * (cs + ad) * idJ
    @inbounds for f in 1:6
        bdry[f, e] == 0 || continue
        a_idx = (f + 1) ÷ 2; row = isodd(f) ? 1 : N
        on = a_idx == 1 ? (i == row) : a_idx == 2 ? (j == row) : (k == row)
        on || continue
        p_, q_ = a_idx == 1 ? (j, k) : a_idx == 2 ? (i, k) : (i, j)
        nbr = Int(neighbour[f, e]); nf = Int(nbr_face[f, e]); o = orient[f, e]
        s_f = isodd(f) ? -one(T) : one(T)
        pn, qn = _neigh_pq(o, p_, q_, Int32(N))
        axc = a_idx == 1 ? ax1[i,j,k,e] : a_idx == 2 ? ax2[i,j,k,e] : ax3[i,j,k,e]
        ayc = a_idx == 1 ? ay1[i,j,k,e] : a_idx == 2 ? ay2[i,j,k,e] : ay3[i,j,k,e]
        azc = a_idx == 1 ? az1[i,j,k,e] : a_idx == 2 ? az2[i,j,k,e] : az3[i,j,k,e]
        Fn_self = s_f * (axc*F1l[i,j,k] + ayc*F2l[i,j,k] + azc*F3l[i,j,k])
        Fn_nbr  = s_f * (axc*ft[pn, qn, nf, nbr, c] +
                         ayc*ft[pn, qn, nf, nbr, C + c] +
                         azc*ft[pn, qn, nf, nbr, 2C + c])
        r += idJ * half * (Fn_nbr - Fn_self) / ops.H[row, row]
    end
    if HASADD
        @inbounds divF[i, j, k, e, c] = r + add[i, j, k, e, c]
    else
        @inbounds divF[i, j, k, e, c] = r
    end
end
