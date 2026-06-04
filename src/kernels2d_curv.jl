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
# BCs are layered downstream). CPU only for now.

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

# (i,j) of face node `p` on face `f` (1,2 → normal axis ξ; 3,4 → η),
# at row `isodd(f) ? 1 : N`.
@inline function _facenode2d(f, p, ::Val{N}) where {N}
    row = isodd(f) ? 1 : N
    return f ≤ 2 ? (row, p) : (p, row)
end
@inline _facesign2d(f, ::Type{T}) where {T} = isodd(f) ? -one(T) : one(T)

"""
    apply_gradient2d!(g1, g2, Φ; geom, ops, metric) → (g1, g2)

Physical gradient `(∂_xΦ, ∂_yΦ)` of `Φ::(N,N,Ne)` on a curvilinear
2D mesh, conservative free-stream-preserving form + centred-flux SAT.
"""
function apply_gradient2d!(g1::AbstractArray{T,3}, g2::AbstractArray{T,3},
                           Φ::AbstractArray{T,3};
                           geom::MeshGeometry{2,T,N}, ops::SBPOps{N,T},
                           metric) where {T,N}
    Ne = geom.Ne; G = ops.G; conn = geom.conn
    H1 = SVector{N,T}(ntuple(i -> ops.H[i, i], Val(N)))
    (; ax1, ax2, ay1, ay2, invdetJ) = metric
    half = one(T) / 2
    @inbounds for e in 1:Ne
        # Volume, SPLIT (skew-symmetric) form ½(conservative + advective):
        #   conservative_a = Σ_α D̂_α(aₐ^α Φ)
        #   advective_a    = Σ_α aₐ^α (D̂_α Φ)
        for j in 1:N, i in 1:N
            csx = zero(T); csy = zero(T); gξ = zero(T); gη = zero(T)
            for p in 1:N
                Gip = G[i, p]; Gjp = G[j, p]
                csx += Gip * ax1[p,j,e] * Φ[p,j,e] + Gjp * ax2[i,p,e] * Φ[i,p,e]
                csy += Gip * ay1[p,j,e] * Φ[p,j,e] + Gjp * ay2[i,p,e] * Φ[i,p,e]
                gξ  += Gip * Φ[p,j,e]
                gη  += Gjp * Φ[i,p,e]
            end
            adx = ax1[i,j,e]*gξ + ax2[i,j,e]*gη
            ady = ay1[i,j,e]*gξ + ay2[i,j,e]*gη
            g1[i,j,e] = half * (csx + adx) * invdetJ[i,j,e]
            g2[i,j,e] = half * (csy + ady) * invdetJ[i,j,e]
        end
        # SAT: lift ½(Φ_nbr − Φ_self) via the self metric term.
        for f in 1:4
            conn.bdry[f, e] == 0 || continue
            nbr = Int(conn.neighbour[f, e]); nf = Int(conn.neighbour_face[f, e])
            o = Int(conn.orientation[f, e]); s_f = _facesign2d(f, T)
            row = isodd(f) ? 1 : N
            for p in 1:N
                ci, cj = _facenode2d(f, p, Val(N))
                pn = _neigh_p(o, p, N); ni, nj = _facenode2d(nf, pn, Val(N))
                jump = Φ[ni,nj,nbr] - Φ[ci,cj,e]
                nfx = s_f * (f ≤ 2 ? ax1[ci,cj,e] : ax2[ci,cj,e])
                nfy = s_f * (f ≤ 2 ? ay1[ci,cj,e] : ay2[ci,cj,e])
                c = invdetJ[ci,cj,e] * half * jump / H1[row]
                g1[ci,cj,e] += c * nfx
                g2[ci,cj,e] += c * nfy
            end
        end
    end
    return g1, g2
end

"""
    apply_divergence2d!(divF, F1, F2; geom, ops, metric) → divF

Physical divergence `∂_xF^x + ∂_yF^y` of the vector field
`(F1, F2)::(N,N,Ne)` on a curvilinear 2D mesh — the SBP-adjoint of
`apply_gradient2d!`, conservative form + centred-flux SAT.
"""
function apply_divergence2d!(divF::AbstractArray{T,3},
                             F1::AbstractArray{T,3}, F2::AbstractArray{T,3};
                             geom::MeshGeometry{2,T,N}, ops::SBPOps{N,T},
                             metric) where {T,N}
    Ne = geom.Ne; G = ops.G; conn = geom.conn
    H1 = SVector{N,T}(ntuple(i -> ops.H[i, i], Val(N)))
    (; ax1, ax2, ay1, ay2, invdetJ) = metric
    half = one(T) / 2
    @inbounds for e in 1:Ne
        # Volume, SPLIT form ½(conservative + advective):
        #   conservative = Σ_α D̂_α(F̃^α),   F̃^α = Σ_a aₐ^α F^a
        #   advective    = Σ_a Σ_α aₐ^α (D̂_α F^a)
        for j in 1:N, i in 1:N
            cs = zero(T)
            gξF1 = zero(T); gηF1 = zero(T); gξF2 = zero(T); gηF2 = zero(T)
            for p in 1:N
                Gip = G[i,p]; Gjp = G[j,p]
                Ft1 = ax1[p,j,e]*F1[p,j,e] + ay1[p,j,e]*F2[p,j,e]   # F̃^1 at (p,j)
                Ft2 = ax2[i,p,e]*F1[i,p,e] + ay2[i,p,e]*F2[i,p,e]   # F̃^2 at (i,p)
                cs += Gip*Ft1 + Gjp*Ft2
                gξF1 += Gip*F1[p,j,e]; gηF1 += Gjp*F1[i,p,e]
                gξF2 += Gip*F2[p,j,e]; gηF2 += Gjp*F2[i,p,e]
            end
            ad = ax1[i,j,e]*gξF1 + ax2[i,j,e]*gηF1 +
                 ay1[i,j,e]*gξF2 + ay2[i,j,e]*gηF2
            divF[i,j,e] = half * (cs + ad) * invdetJ[i,j,e]
        end
        # SAT: centred-flux of the physical normal flux (self metric).
        for f in 1:4
            conn.bdry[f, e] == 0 || continue
            nbr = Int(conn.neighbour[f, e]); nf = Int(conn.neighbour_face[f, e])
            o = Int(conn.orientation[f, e]); s_f = _facesign2d(f, T)
            row = isodd(f) ? 1 : N
            for p in 1:N
                ci, cj = _facenode2d(f, p, Val(N))
                pn = _neigh_p(o, p, N); ni, nj = _facenode2d(nf, pn, Val(N))
                nfx = s_f * (f ≤ 2 ? ax1[ci,cj,e] : ax2[ci,cj,e])
                nfy = s_f * (f ≤ 2 ? ay1[ci,cj,e] : ay2[ci,cj,e])
                Fn_self = nfx*F1[ci,cj,e] + nfy*F2[ci,cj,e]
                Fn_nbr  = nfx*F1[ni,nj,nbr] + nfy*F2[ni,nj,nbr]
                divF[ci,cj,e] += invdetJ[ci,cj,e] * half * (Fn_nbr - Fn_self) / H1[row]
            end
        end
    end
    return divF
end
