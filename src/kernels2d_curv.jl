# Free-stream-preserving conservative first-derivative (gradient /
# divergence) for CURVILINEAR 2D meshes — the building block for the
# conservative scalar wave on cubed-square / inflated-square meshes.
#
# Metric terms are computed DISCRETELY from the nodal coordinates
# (`make_geometry` stores analytic Jacobians, which do not satisfy the
# discrete metric identities). With discrete metric terms the 2D
# identities Σ_α D̂_α(aₐ^α) = 0 hold automatically (D̂_ξ D̂_η Y −
# D̂_η D̂_ξ Y = 0: tensor-product SBP operators commute), so the
# conservative form is free-stream-preserving by construction.
#
# Conventions: fields are (N, N, Ne) with ξ = axis i, η = axis j.
# Metric terms aₐ^α = detJ·∂ξ_α/∂x_a:  ax1=Yη, ax2=−Yξ, ay1=−Xη, ay2=Xξ.
#   Gradient   ∂_aΦ   = invdetJ·[ D̂_ξ(aₐ^1 Φ) + D̂_η(aₐ^2 Φ) ] + SAT
#   Divergence ∂_aF^a = invdetJ·[ D̂_ξ F̃^1 + D̂_η F̃^2 ] + SAT,
#              F̃^α = Σ_a aₐ^α F^a
# SAT (both): centred-flux lift of the PHYSICAL field jump at interior
# faces, weighted by the self metric term — the exactly-skew,
# free-stream-preserving structure of the 1D `apply_D!`. The metric
# term equals the outward normal × surface element, so the jump of a
# constant field (zero) and the GCL (zero metric divergence) together
# give free-stream. `bdry ≠ 0` faces get no SAT (one-sided/excision;
# BCs are layered downstream).
#
# Structure: a SINGLE set of KernelAbstractions kernels runs on both CPU
# and GPU (the CPU backend executes them serially; it is used to test the
# GPU path — highest CPU efficiency is not a goal). Each operator is
# TWO passes, mirroring `apply_laplacian!`:
#   1. gather  — every element writes its face-node field values into the
#      workspace face trace `work.face_trace[ch, p, f, e]` (gradient: Φ in
#      channel 1; divergence: F1, F2 in channels 1, 2);
#   2. volume+SAT — workgroup-per-element, N² workitems; stage the field
#      into `@localmem`, do the split-form volume reduction, then apply the
#      centred-flux SAT by reading the NEIGHBOUR's gathered trace
#      `work.face_trace[ch, pn, nbr_face, nbr]` (orientation via `_neigh_p`).
#      Each output node is written by exactly one workitem (gather, not
#      scatter) — no races. The two separate launches give the global
#      barrier between gather and read.

"""
    make_metric_terms2d(geom::MeshGeometry{2,T,N}, ops) → NamedTuple

Discrete metric terms `(; ax1, ax2, ay1, ay2, invdetJ, Hd)` (each
`(N,N,Ne)`) computed from `geom.coords` via `ops.G`. `Hd` is the
discrete mass `H_1d[i]·H_1d[j]·|detJ|`, the energy norm consistent with
the operator (use instead of `geom.Hphys`).
"""
function make_metric_terms2d(geom::MeshGeometry{2, T, N}, ops::SBPOps{N, T}) where {T, N}
    Ne = geom.Ne; G = ops.G
    H1 = SVector{N,T}(ntuple(i -> ops.H[i, i], Val(N)))
    ax1 = Array{T,3}(undef, N, N, Ne); ax2 = similar(ax1)
    ay1 = similar(ax1); ay2 = similar(ax1)
    invdetJ = similar(ax1); Hd = similar(ax1)
    @inbounds for e in 1:Ne, j in 1:N, i in 1:N
        Xξ = zero(T); Xη = zero(T); Yξ = zero(T); Yη = zero(T)
        for p in 1:N
            Xξ += G[i, p] * geom.coords[1, p, j, e]
            Yξ += G[i, p] * geom.coords[2, p, j, e]
            Xη += G[j, p] * geom.coords[1, i, p, e]
            Yη += G[j, p] * geom.coords[2, i, p, e]
        end
        detJ = Xξ * Yη - Xη * Yξ
        ax1[i, j, e] =  Yη; ax2[i, j, e] = -Yξ
        ay1[i, j, e] = -Xη; ay2[i, j, e] =  Xξ
        invdetJ[i, j, e] = one(T) / detJ
        Hd[i, j, e] = H1[i] * H1[j] * abs(detJ)
    end
    return (; ax1, ax2, ay1, ay2, invdetJ, Hd)
end

"""
    metric_to_device(metric, backend) → NamedTuple

Migrate the discrete metric-term bundle from [`make_metric_terms2d`]
(`ax1, ax2, ay1, ay2, invdetJ, Hd`) onto `backend` (the curvilinear
operators read these on-device). `make_metric_terms2d` runs on the
HOST geom (a scalar nodal loop), so a GPU caller computes the terms on
the host geom and then migrates them with this helper.
"""
function metric_to_device(metric, backend)
    _mv(a) = (d = KernelAbstractions.allocate(backend, eltype(a), size(a));
              copyto!(d, a); d)
    return (; ax1 = _mv(metric.ax1), ax2 = _mv(metric.ax2),
            ay1 = _mv(metric.ay1), ay2 = _mv(metric.ay2),
            invdetJ = _mv(metric.invdetJ), Hd = _mv(metric.Hd))
end

# (i,j) of face node `p` on face `f` (1,2 → normal axis ξ; 3,4 → η),
# at row `isodd(f) ? 1 : N`.
@inline function _facenode2d(f, p, ::Val{N}) where {N}
    row = isodd(f) ? 1 : N
    return f ≤ 2 ? (row, p) : (p, row)
end
@inline _facesign2d(f, ::Type{T}) where {T} = isodd(f) ? -one(T) : one(T)

"""
    apply_gradient2d!(g1, g2, Φ; geom, ops, metric, work) → (g1, g2)

Physical gradient `(∂_xΦ, ∂_yΦ)` of `Φ::(N,N,Ne)` on a curvilinear
2D mesh, conservative free-stream-preserving form + centred-flux SAT.
Two KernelAbstractions passes (gather → volume+SAT) that run on both
CPU and GPU; `work::MeshWorkspace{2,T,N}` supplies the face-trace
buffer (channel 1 holds Φ at face nodes).
"""
function apply_gradient2d!(g1::AbstractArray{T,3}, g2::AbstractArray{T,3},
                           Φ::AbstractArray{T,3};
                           geom::MeshGeometry{2,T,N}, ops::SBPOps{N,T},
                           metric, work::MeshWorkspace{2,T,N}) where {T,N}
    backend = get_backend(Φ)
    _gather_face2d_1ch!(backend, (N, N))(
        Φ, work, geom.conn.bdry, Val(N); ndrange = (N, N, geom.Ne))
    _grad2d_volume_kernel!(backend, N^2)(
        g1, g2, Φ, work, ops, metric.ax1, metric.ax2, metric.ay1, metric.ay2,
        metric.invdetJ, geom.conn.neighbour, geom.conn.neighbour_face,
        geom.conn.orientation, geom.conn.bdry, Val(N);
        ndrange = N^2 * geom.Ne)
    return g1, g2
end

# Pass 1 (gather) is the shared `_gather_face2d_1ch!` (Φ → channel 1),
# defined in kernels2d_grad.jl.
#
# Pass 2: per element, stage Φ into `@localmem`, do the split-form volume
# reduction, then apply the centred-flux SAT reading the neighbour's
# gathered trace. One write per node (gather, no scatter).
@kernel function _grad2d_volume_kernel!(g1::AbstractArray{T}, g2,
                                        @Const(Φ), work, ops,
                                        @Const(ax1), @Const(ax2), @Const(ay1),
                                        @Const(ay2), @Const(invdetJ),
                                        @Const(neighbour), @Const(nbr_face),
                                        @Const(orient), @Const(bdry),
                                        ::Val{N}) where {T, N}
    e    = @index(Group, Linear)
    li   = @index(Local, Linear)
    i, j = _ij_from_li(li, Val(N))
    u_loc = @localmem T (N, N)
    @inbounds u_loc[i, j] = Φ[i, j, e]
    @synchronize
    e    = @index(Group, Linear)
    li   = @index(Local, Linear)
    i, j = _ij_from_li(li, Val(N))
    G = ops.G; half = T(1) / 2
    csx = zero(T); csy = zero(T); gξ = zero(T); gη = zero(T)
    @inbounds for p in 1:N
        Gip = G[i, p]; Gjp = G[j, p]
        csx += Gip*ax1[p,j,e]*u_loc[p,j] + Gjp*ax2[i,p,e]*u_loc[i,p]
        csy += Gip*ay1[p,j,e]*u_loc[p,j] + Gjp*ay2[i,p,e]*u_loc[i,p]
        gξ  += Gip*u_loc[p,j]; gη += Gjp*u_loc[i,p]
    end
    @inbounds idJ = invdetJ[i,j,e]
    @inbounds r1 = half*(csx + ax1[i,j,e]*gξ + ax2[i,j,e]*gη)*idJ
    @inbounds r2 = half*(csy + ay1[i,j,e]*gξ + ay2[i,j,e]*gη)*idJ
    @inbounds for f in 1:4
        bdry[f,e] == 0 || continue
        a_idx = (f + 1) ÷ 2; row = isodd(f) ? 1 : N
        on = a_idx == 1 ? (i == row) : (j == row)
        on || continue
        p_local = a_idx == 1 ? j : i
        nbr = Int(neighbour[f,e]); nf = Int(nbr_face[f,e]); o = Int(orient[f,e])
        s_f = isodd(f) ? -one(T) : one(T)
        pn = _neigh_p(o, p_local, N)
        jump = work.face_trace[1, pn, nf, nbr] - u_loc[i,j]
        nfx = s_f * (f ≤ 2 ? ax1[i,j,e] : ax2[i,j,e])
        nfy = s_f * (f ≤ 2 ? ay1[i,j,e] : ay2[i,j,e])
        c = idJ * half * jump / ops.H[row, row]
        r1 += c*nfx; r2 += c*nfy
    end
    @inbounds g1[i,j,e] = r1
    @inbounds g2[i,j,e] = r2
end

"""
    apply_divergence2d!(divF, F1, F2; geom, ops, metric, work) → divF

Physical divergence `∂_xF^x + ∂_yF^y` of the vector field
`(F1, F2)::(N,N,Ne)` on a curvilinear 2D mesh — the SBP-adjoint of
`apply_gradient2d!`, conservative form + centred-flux SAT. Two KA
passes; `work.face_trace` channels 1, 2 hold F1, F2 at face nodes.
"""
function apply_divergence2d!(divF::AbstractArray{T,3},
                             F1::AbstractArray{T,3}, F2::AbstractArray{T,3};
                             geom::MeshGeometry{2,T,N}, ops::SBPOps{N,T},
                             metric, work::MeshWorkspace{2,T,N}) where {T,N}
    backend = get_backend(divF)
    _gather_face2d_2ch!(backend, (N, N))(
        F1, F2, work, geom.conn.bdry, Val(N); ndrange = (N, N, geom.Ne))
    _div2d_volume_kernel!(backend, N^2)(
        divF, F1, F2, work, ops, metric.ax1, metric.ax2, metric.ay1, metric.ay2,
        metric.invdetJ, geom.conn.neighbour, geom.conn.neighbour_face,
        geom.conn.orientation, geom.conn.bdry, Val(N);
        ndrange = N^2 * geom.Ne)
    return divF
end

# Pass 1 (gather) is the shared `_gather_face2d_2ch!` (F1, F2 → channels
# 1, 2), defined in kernels2d_grad.jl.
#
# Pass 2: stage (F1, F2) into `@localmem`, split-form volume divergence,
# then centred-flux SAT reading the neighbour's gathered (F1, F2).
@kernel function _div2d_volume_kernel!(divF::AbstractArray{T},
                                       @Const(F1), @Const(F2), work, ops,
                                       @Const(ax1), @Const(ax2), @Const(ay1),
                                       @Const(ay2), @Const(invdetJ),
                                       @Const(neighbour), @Const(nbr_face),
                                       @Const(orient), @Const(bdry),
                                       ::Val{N}) where {T, N}
    e    = @index(Group, Linear)
    li   = @index(Local, Linear)
    i, j = _ij_from_li(li, Val(N))
    F1_loc = @localmem T (N, N)
    F2_loc = @localmem T (N, N)
    @inbounds F1_loc[i, j] = F1[i, j, e]
    @inbounds F2_loc[i, j] = F2[i, j, e]
    @synchronize
    e    = @index(Group, Linear)
    li   = @index(Local, Linear)
    i, j = _ij_from_li(li, Val(N))
    G = ops.G; half = T(1) / 2
    cs = zero(T); gξF1 = zero(T); gηF1 = zero(T); gξF2 = zero(T); gηF2 = zero(T)
    @inbounds for p in 1:N
        Gip = G[i,p]; Gjp = G[j,p]
        Ft1 = ax1[p,j,e]*F1_loc[p,j] + ay1[p,j,e]*F2_loc[p,j]   # F̃^1 at (p,j)
        Ft2 = ax2[i,p,e]*F1_loc[i,p] + ay2[i,p,e]*F2_loc[i,p]   # F̃^2 at (i,p)
        cs += Gip*Ft1 + Gjp*Ft2
        gξF1 += Gip*F1_loc[p,j]; gηF1 += Gjp*F1_loc[i,p]
        gξF2 += Gip*F2_loc[p,j]; gηF2 += Gjp*F2_loc[i,p]
    end
    @inbounds idJ = invdetJ[i,j,e]
    @inbounds ad = ax1[i,j,e]*gξF1 + ax2[i,j,e]*gηF1 +
                   ay1[i,j,e]*gξF2 + ay2[i,j,e]*gηF2
    r = half*(cs + ad)*idJ
    @inbounds for f in 1:4
        bdry[f,e] == 0 || continue
        a_idx = (f + 1) ÷ 2; row = isodd(f) ? 1 : N
        on = a_idx == 1 ? (i == row) : (j == row)
        on || continue
        p_local = a_idx == 1 ? j : i
        nbr = Int(neighbour[f,e]); nf = Int(nbr_face[f,e]); o = Int(orient[f,e])
        s_f = isodd(f) ? -one(T) : one(T)
        pn = _neigh_p(o, p_local, N)
        nfx = s_f * (f ≤ 2 ? ax1[i,j,e] : ax2[i,j,e])
        nfy = s_f * (f ≤ 2 ? ay1[i,j,e] : ay2[i,j,e])
        Fn_self = nfx*F1_loc[i,j] + nfy*F2_loc[i,j]
        Fn_nbr  = nfx*work.face_trace[1,pn,nf,nbr] + nfy*work.face_trace[2,pn,nf,nbr]
        r += idJ * half * (Fn_nbr - Fn_self) / ops.H[row, row]
    end
    @inbounds divF[i,j,e] = r
end
