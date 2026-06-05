# Free-stream-preserving conservative first-derivative (gradient /
# divergence) for CURVILINEAR 3D meshes — the 3D analog of
# kernels2d_curv.jl. The building block for the conservative scalar wave
# on cubed-cube / inflated-cube / radial-shell meshes.
#
# Unlike 2D (where the discrete metric identities Σ_α D̂_α(aₐ^α)=0 hold
# automatically because tensor-product SBP operators commute), in 3D the
# straightforward metric terms do NOT satisfy the identities. The fix is
# the CONSERVATIVE-CURL form (Thomas–Lombard / Kopriva): the contravariant
# metric J·∂ξ_α/∂x_n = (∇_ξ X_l × ∇_ξ X_m)_α (cross-product form, n,l,m
# cyclic) is rewritten as the curl of a vector potential,
#
#     J a^α_n = (∇_ξ × C_n)_α,   C_n = ½(X_l ∇_ξ X_m − X_m ∇_ξ X_l),
#
# using the IDENTITY ∇×(X_l ∇X_m) = ∇X_l × ∇X_m (curl of a gradient is
# zero). Then the discrete metric divergence Σ_α D̂_α(J a^α_n) =
# D̂·(D̂×C_n) = 0 to round-off, because the 1D SBP derivatives commute
# across reference directions (D̂_β D̂_γ = D̂_γ D̂_β for tensor-product
# operators) — i.e. the discrete divergence of a discrete curl vanishes.
# That is the free-stream guarantee.
#
# Conventions: fields are (N,N,N,Ne), ξ = axis i, η = axis j, ζ = axis k.
# Metric vectors a_n^α = J·∂ξ_α/∂x_n (n ∈ {x,y,z}, α ∈ {ξ,η,ζ}); 9
# components ax1..az3, plus invdetJ and the discrete mass Hd.

"""
    make_metric_terms3d(geom::MeshGeometry{3,T,N}, ops) → NamedTuple

Discrete metric terms `(; ax1,ax2,ax3, ay1,ay2,ay3, az1,az2,az3,
invdetJ, Hd)` (each `(N,N,N,Ne)`) computed from `geom.coords` via the
conservative-curl form, so the discrete metric identities
`Σ_α D̂_α(a_n^α) = 0` hold to round-off (free-stream preservation). `Hd`
is the discrete mass `H_1d[i]·H_1d[j]·H_1d[k]·|detJ|`. Host computation
(scalar nodal loop with `ops.G`); migrate to a device with
[`metric_to_device`](@ref).
"""
function make_metric_terms3d(geom::MeshGeometry{3, T, N}, ops::SBPOps{N, T}) where {T, N}
    Ne = geom.Ne; G = ops.G
    H1 = SVector{N,T}(ntuple(i -> ops.H[i, i], Val(N)))
    mk() = Array{T,4}(undef, N, N, N, Ne)
    ax1=mk(); ax2=mk(); ax3=mk(); ay1=mk(); ay2=mk(); ay3=mk()
    az1=mk(); az2=mk(); az3=mk(); invdetJ=mk(); Hd=mk()

    # Per-element scratch (reused). DX[a,β,i,j,k] = (D̂_β X_a) at the node;
    # C[n,β,i,j,k] = ½(X_l D̂_β X_m − X_m D̂_β X_l) with (n,l,m) cyclic.
    DX = Array{T,5}(undef, 3, 3, N, N, N)
    C  = Array{T,5}(undef, 3, 3, N, N, N)
    cyc = ((2, 3), (3, 1), (1, 2))   # (l, m) for n = 1, 2, 3

    @inbounds for e in 1:Ne
        # 1) Reference derivatives of the coordinates: DX[a,β] = D̂_β x_a.
        for a in 1:3, kk in 1:N, jj in 1:N, ii in 1:N
            sξ = zero(T); sη = zero(T); sζ = zero(T)
            for p in 1:N
                sξ += G[ii, p] * geom.coords[a, p, jj, kk, e]
                sη += G[jj, p] * geom.coords[a, ii, p, kk, e]
                sζ += G[kk, p] * geom.coords[a, ii, jj, p, e]
            end
            DX[a, 1, ii, jj, kk] = sξ
            DX[a, 2, ii, jj, kk] = sη
            DX[a, 3, ii, jj, kk] = sζ
        end
        # detJ + Hd + invdetJ from the nodal Jacobian J[a,β] = DX[a,β].
        for kk in 1:N, jj in 1:N, ii in 1:N
            J11=DX[1,1,ii,jj,kk]; J12=DX[1,2,ii,jj,kk]; J13=DX[1,3,ii,jj,kk]
            J21=DX[2,1,ii,jj,kk]; J22=DX[2,2,ii,jj,kk]; J23=DX[2,3,ii,jj,kk]
            J31=DX[3,1,ii,jj,kk]; J32=DX[3,2,ii,jj,kk]; J33=DX[3,3,ii,jj,kk]
            dJ = J11*(J22*J33 - J23*J32) - J12*(J21*J33 - J23*J31) +
                 J13*(J21*J32 - J22*J31)
            invdetJ[ii,jj,kk,e] = one(T) / dJ
            Hd[ii,jj,kk,e] = H1[ii]*H1[jj]*H1[kk]*abs(dJ)
        end
        # 2) Vector potential C[n,β] = ½(X_l D̂_β X_m − X_m D̂_β X_l).
        for n in 1:3
            l, m = cyc[n]
            for kk in 1:N, jj in 1:N, ii in 1:N
                Xl = geom.coords[l, ii, jj, kk, e]
                Xm = geom.coords[m, ii, jj, kk, e]
                for β in 1:3
                    C[n, β, ii, jj, kk] =
                        (Xl * DX[m, β, ii, jj, kk] - Xm * DX[l, β, ii, jj, kk]) / 2
                end
            end
        end
        # 3) a_n^α = (D̂ × C_n)_α  (outer SBP-G curl):
        #    a_n^ξ = D̂_η C_n^ζ − D̂_ζ C_n^η
        #    a_n^η = D̂_ζ C_n^ξ − D̂_ξ C_n^ζ
        #    a_n^ζ = D̂_ξ C_n^η − D̂_η C_n^ξ
        for n in 1:3
            ax = n == 1 ? ax1 : n == 2 ? ay1 : az1
            ay = n == 1 ? ax2 : n == 2 ? ay2 : az2
            az = n == 1 ? ax3 : n == 2 ? ay3 : az3
            for kk in 1:N, jj in 1:N, ii in 1:N
                # D̂_ξ of C[n,·]; D̂_η; D̂_ζ — only the components we need.
                dη_Cζ = zero(T); dζ_Cη = zero(T)   # for ξ-component
                dζ_Cξ = zero(T); dξ_Cζ = zero(T)   # for η-component
                dξ_Cη = zero(T); dη_Cξ = zero(T)   # for ζ-component
                for p in 1:N
                    Gi = G[ii, p]; Gj = G[jj, p]; Gk = G[kk, p]
                    dη_Cζ += Gj * C[n, 3, ii, p, kk]
                    dζ_Cη += Gk * C[n, 2, ii, jj, p]
                    dζ_Cξ += Gk * C[n, 1, ii, jj, p]
                    dξ_Cζ += Gi * C[n, 3, p, jj, kk]
                    dξ_Cη += Gi * C[n, 2, p, jj, kk]
                    dη_Cξ += Gj * C[n, 1, ii, p, kk]
                end
                ax[ii,jj,kk,e] = dη_Cζ - dζ_Cη
                ay[ii,jj,kk,e] = dζ_Cξ - dξ_Cζ
                az[ii,jj,kk,e] = dξ_Cη - dη_Cξ
            end
        end
    end
    return (; ax1, ax2, ax3, ay1, ay2, ay3, az1, az2, az3, invdetJ, Hd)
end

# ---- Shared 3-channel face gather (pass 1 of the divergence) ----
# Write each element's interior-face (F1,F2,F3) into work.face_trace
# channels 1,2,3, following the Laplacian tangential convention (face
# normal to axis 1 → (p,q)=(j,k); axis 2 → (i,k); axis 3 → (i,j)). The
# gradient reuses the 1-channel `_gather_face3d_1ch!` (kernels3d_grad.jl).
@kernel function _gather_face3d_3ch!(@Const(F1), @Const(F2), @Const(F3),
                                     work, @Const(bdry), ::Val{N}) where {N}
    i, j, k, e = @index(Global, NTuple)
    ft = work.face_trace
    @inbounds begin
        v1 = F1[i,j,k,e]; v2 = F2[i,j,k,e]; v3 = F3[i,j,k,e]
        if i == 1 && bdry[1,e] == 0; ft[1,j,k,1,e]=v1; ft[2,j,k,1,e]=v2; ft[3,j,k,1,e]=v3; end
        if i == N && bdry[2,e] == 0; ft[1,j,k,2,e]=v1; ft[2,j,k,2,e]=v2; ft[3,j,k,2,e]=v3; end
        if j == 1 && bdry[3,e] == 0; ft[1,i,k,3,e]=v1; ft[2,i,k,3,e]=v2; ft[3,i,k,3,e]=v3; end
        if j == N && bdry[4,e] == 0; ft[1,i,k,4,e]=v1; ft[2,i,k,4,e]=v2; ft[3,i,k,4,e]=v3; end
        if k == 1 && bdry[5,e] == 0; ft[1,i,j,5,e]=v1; ft[2,i,j,5,e]=v2; ft[3,i,j,5,e]=v3; end
        if k == N && bdry[6,e] == 0; ft[1,i,j,6,e]=v1; ft[2,i,j,6,e]=v2; ft[3,i,j,6,e]=v3; end
    end
end

"""
    apply_gradient3d!(g1, g2, g3, Φ; geom, ops, metric, work) → (g1, g2, g3)

Physical gradient `(∂_xΦ, ∂_yΦ, ∂_zΦ)` of `Φ::(N,N,N,Ne)` on a
curvilinear 3D mesh, conservative free-stream-preserving split form +
centred-flux SAT (conservative-curl metric from `make_metric_terms3d`).
Two KA passes (gather → volume+SAT) on both CPU and GPU.
"""
function apply_gradient3d!(g1::AbstractArray{T,4}, g2::AbstractArray{T,4},
                           g3::AbstractArray{T,4}, Φ::AbstractArray{T,4};
                           geom::MeshGeometry{3,T,N}, ops::SBPOps{N,T},
                           metric, work::MeshWorkspace{3,T,N}) where {T,N}
    backend = get_backend(Φ)
    _gather_face3d_1ch!(backend, (N,N,N))(
        Φ, work, geom.conn.bdry, Val(N); ndrange = (N,N,N,geom.Ne))
    _grad3d_volume_kernel!(backend, N^3)(
        g1, g2, g3, Φ, work, ops,
        metric.ax1, metric.ax2, metric.ax3, metric.ay1, metric.ay2, metric.ay3,
        metric.az1, metric.az2, metric.az3, metric.invdetJ,
        geom.conn.neighbour, geom.conn.neighbour_face, geom.conn.orientation,
        geom.conn.bdry, Val(N); ndrange = N^3 * geom.Ne)
    return g1, g2, g3
end

@kernel function _grad3d_volume_kernel!(g1::AbstractArray{T}, g2, g3,
        @Const(Φ), work, ops, @Const(ax1), @Const(ax2), @Const(ax3),
        @Const(ay1), @Const(ay2), @Const(ay3), @Const(az1), @Const(az2),
        @Const(az3), @Const(invdetJ), @Const(neighbour), @Const(nbr_face),
        @Const(orient), @Const(bdry), ::Val{N}) where {T, N}
    e = @index(Group, Linear); li = @index(Local, Linear)
    i, j, k = _ijk_from_li(li, Val(N))
    u_loc = @localmem T (N, N, N)
    @inbounds u_loc[i,j,k] = Φ[i,j,k,e]
    @synchronize
    e = @index(Group, Linear); li = @index(Local, Linear)
    i, j, k = _ijk_from_li(li, Val(N))
    G = ops.G; half = T(1)/2
    csx=zero(T); csy=zero(T); csz=zero(T); gξ=zero(T); gη=zero(T); gζ=zero(T)
    @inbounds for p in 1:N
        Gi=G[i,p]; Gj=G[j,p]; Gk=G[k,p]
        up=u_loc[p,j,k]; uq=u_loc[i,p,k]; ur=u_loc[i,j,p]
        csx += Gi*ax1[p,j,k,e]*up + Gj*ax2[i,p,k,e]*uq + Gk*ax3[i,j,p,e]*ur
        csy += Gi*ay1[p,j,k,e]*up + Gj*ay2[i,p,k,e]*uq + Gk*ay3[i,j,p,e]*ur
        csz += Gi*az1[p,j,k,e]*up + Gj*az2[i,p,k,e]*uq + Gk*az3[i,j,p,e]*ur
        gξ += Gi*up; gη += Gj*uq; gζ += Gk*ur
    end
    @inbounds idJ = invdetJ[i,j,k,e]
    @inbounds r1 = half*(csx + ax1[i,j,k,e]*gξ + ax2[i,j,k,e]*gη + ax3[i,j,k,e]*gζ)*idJ
    @inbounds r2 = half*(csy + ay1[i,j,k,e]*gξ + ay2[i,j,k,e]*gη + ay3[i,j,k,e]*gζ)*idJ
    @inbounds r3 = half*(csz + az1[i,j,k,e]*gξ + az2[i,j,k,e]*gη + az3[i,j,k,e]*gζ)*idJ
    @inbounds for f in 1:6
        bdry[f,e] == 0 || continue
        a_idx = (f+1)÷2; row = isodd(f) ? 1 : N
        on = a_idx==1 ? (i==row) : a_idx==2 ? (j==row) : (k==row)
        on || continue
        p_, q_ = a_idx==1 ? (j,k) : a_idx==2 ? (i,k) : (i,j)
        nbr=Int(neighbour[f,e]); nf=Int(nbr_face[f,e]); o=orient[f,e]
        s_f = isodd(f) ? -one(T) : one(T)
        pn, qn = _neigh_pq(o, p_, q_, Int32(N))
        jump = work.face_trace[1,pn,qn,nf,nbr] - u_loc[i,j,k]
        axc = a_idx==1 ? ax1[i,j,k,e] : a_idx==2 ? ax2[i,j,k,e] : ax3[i,j,k,e]
        ayc = a_idx==1 ? ay1[i,j,k,e] : a_idx==2 ? ay2[i,j,k,e] : ay3[i,j,k,e]
        azc = a_idx==1 ? az1[i,j,k,e] : a_idx==2 ? az2[i,j,k,e] : az3[i,j,k,e]
        c = idJ*half*jump/ops.H[row,row]
        r1 += c*s_f*axc; r2 += c*s_f*ayc; r3 += c*s_f*azc
    end
    @inbounds g1[i,j,k,e]=r1; @inbounds g2[i,j,k,e]=r2; @inbounds g3[i,j,k,e]=r3
end

"""
    apply_divergence3d!(divF, F1, F2, F3; geom, ops, metric, work) → divF

Physical divergence `∂_xF^x+∂_yF^y+∂_zF^z` on a curvilinear 3D mesh —
the SBP-adjoint of `apply_gradient3d!`, conservative split form +
centred-flux SAT. `work.face_trace` channels 1,2,3 hold F1,F2,F3.
"""
function apply_divergence3d!(divF::AbstractArray{T,4}, F1::AbstractArray{T,4},
                             F2::AbstractArray{T,4}, F3::AbstractArray{T,4};
                             geom::MeshGeometry{3,T,N}, ops::SBPOps{N,T},
                             metric, work::MeshWorkspace{3,T,N}) where {T,N}
    backend = get_backend(divF)
    _gather_face3d_3ch!(backend, (N,N,N))(
        F1, F2, F3, work, geom.conn.bdry, Val(N); ndrange = (N,N,N,geom.Ne))
    _div3d_volume_kernel!(backend, N^3)(
        divF, F1, F2, F3, work, ops,
        metric.ax1, metric.ax2, metric.ax3, metric.ay1, metric.ay2, metric.ay3,
        metric.az1, metric.az2, metric.az3, metric.invdetJ,
        geom.conn.neighbour, geom.conn.neighbour_face, geom.conn.orientation,
        geom.conn.bdry, Val(N); ndrange = N^3 * geom.Ne)
    return divF
end

@kernel function _div3d_volume_kernel!(divF::AbstractArray{T}, @Const(F1),
        @Const(F2), @Const(F3), work, ops, @Const(ax1), @Const(ax2),
        @Const(ax3), @Const(ay1), @Const(ay2), @Const(ay3), @Const(az1),
        @Const(az2), @Const(az3), @Const(invdetJ), @Const(neighbour),
        @Const(nbr_face), @Const(orient), @Const(bdry), ::Val{N}) where {T, N}
    e = @index(Group, Linear); li = @index(Local, Linear)
    i, j, k = _ijk_from_li(li, Val(N))
    F1l = @localmem T (N,N,N); F2l = @localmem T (N,N,N); F3l = @localmem T (N,N,N)
    @inbounds F1l[i,j,k]=F1[i,j,k,e]; @inbounds F2l[i,j,k]=F2[i,j,k,e]
    @inbounds F3l[i,j,k]=F3[i,j,k,e]
    @synchronize
    e = @index(Group, Linear); li = @index(Local, Linear)
    i, j, k = _ijk_from_li(li, Val(N))
    G = ops.G; half = T(1)/2
    cs=zero(T)
    gξ1=zero(T);gη1=zero(T);gζ1=zero(T); gξ2=zero(T);gη2=zero(T);gζ2=zero(T)
    gξ3=zero(T);gη3=zero(T);gζ3=zero(T)
    @inbounds for p in 1:N
        Gi=G[i,p]; Gj=G[j,p]; Gk=G[k,p]
        # F̃^ξ at (p,j,k), F̃^η at (i,p,k), F̃^ζ at (i,j,p)
        Fξ = ax1[p,j,k,e]*F1l[p,j,k] + ay1[p,j,k,e]*F2l[p,j,k] + az1[p,j,k,e]*F3l[p,j,k]
        Fη = ax2[i,p,k,e]*F1l[i,p,k] + ay2[i,p,k,e]*F2l[i,p,k] + az2[i,p,k,e]*F3l[i,p,k]
        Fζ = ax3[i,j,p,e]*F1l[i,j,p] + ay3[i,j,p,e]*F2l[i,j,p] + az3[i,j,p,e]*F3l[i,j,p]
        cs += Gi*Fξ + Gj*Fη + Gk*Fζ
        gξ1+=Gi*F1l[p,j,k]; gη1+=Gj*F1l[i,p,k]; gζ1+=Gk*F1l[i,j,p]
        gξ2+=Gi*F2l[p,j,k]; gη2+=Gj*F2l[i,p,k]; gζ2+=Gk*F2l[i,j,p]
        gξ3+=Gi*F3l[p,j,k]; gη3+=Gj*F3l[i,p,k]; gζ3+=Gk*F3l[i,j,p]
    end
    @inbounds idJ = invdetJ[i,j,k,e]
    @inbounds ad = ax1[i,j,k,e]*gξ1 + ax2[i,j,k,e]*gη1 + ax3[i,j,k,e]*gζ1 +
                   ay1[i,j,k,e]*gξ2 + ay2[i,j,k,e]*gη2 + ay3[i,j,k,e]*gζ2 +
                   az1[i,j,k,e]*gξ3 + az2[i,j,k,e]*gη3 + az3[i,j,k,e]*gζ3
    r = half*(cs + ad)*idJ
    @inbounds for f in 1:6
        bdry[f,e] == 0 || continue
        a_idx = (f+1)÷2; row = isodd(f) ? 1 : N
        on = a_idx==1 ? (i==row) : a_idx==2 ? (j==row) : (k==row)
        on || continue
        p_, q_ = a_idx==1 ? (j,k) : a_idx==2 ? (i,k) : (i,j)
        nbr=Int(neighbour[f,e]); nf=Int(nbr_face[f,e]); o=orient[f,e]
        s_f = isodd(f) ? -one(T) : one(T)
        pn, qn = _neigh_pq(o, p_, q_, Int32(N))
        axc = a_idx==1 ? ax1[i,j,k,e] : a_idx==2 ? ax2[i,j,k,e] : ax3[i,j,k,e]
        ayc = a_idx==1 ? ay1[i,j,k,e] : a_idx==2 ? ay2[i,j,k,e] : ay3[i,j,k,e]
        azc = a_idx==1 ? az1[i,j,k,e] : a_idx==2 ? az2[i,j,k,e] : az3[i,j,k,e]
        Fn_self = s_f*(axc*F1l[i,j,k] + ayc*F2l[i,j,k] + azc*F3l[i,j,k])
        Fn_nbr  = s_f*(axc*work.face_trace[1,pn,qn,nf,nbr] +
                       ayc*work.face_trace[2,pn,qn,nf,nbr] +
                       azc*work.face_trace[3,pn,qn,nf,nbr])
        r += idJ*half*(Fn_nbr - Fn_self)/ops.H[row,row]
    end
    @inbounds divF[i,j,k,e] = r
end
