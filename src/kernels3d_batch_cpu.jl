# Plain-Julia threaded CPU fast path for the channel-batched operators.
#
# The KernelAbstractions CPU backend executes the batch kernels one
# node per workitem: a scalar N-point dot with no SIMD, a localmem
# staging pass split by `@synchronize`, a stride-9 `invjac[d, d, …]`
# gather, and branchy per-node SAT handling — measured at ~20 GB/s
# effective on a 2-socket EPYC versus ~300 GB/s STREAM, i.e.
# instruction-bound by an order of magnitude. The loops here process
# one `(element, channel)` block per iteration with SIMD across
# *output* nodes, read the contiguous per-axis Jacobian diagonal
# `geom.invjacd`, and read neighbour face values directly from the
# neighbour's element block (no gather pass, no face-trace traffic).
#
# CPU-SIMD wants these pencil-shaped loops while GPU coalescing wants
# the node-per-thread KA kernels — no single loop structure is fast on
# both, so the two paths coexist: KA remains the GPU implementation,
# and the test suite pins them together BITWISE.
#
# Bitwise-agreement rules (the tests assert `==`, not `≈`):
#   * never put `@simd` / `muladd` / `@fastmath` on the `p`-reduction —
#     SIMD lanes only across outputs (`i`, or the `(i, j)` plane);
#   * keep the kernels' exact expression trees, e.g.
#     `(u_self − u_nbr) * half * idJ / H[1, 1]` (no reciprocal
#     hoisting, no coefficient prefusion);
#   * fold the SAT into the element buffer BEFORE the single fused
#     `scale`/accumulate writeback;
#   * `invjacd` is a verbatim per-node copy of `invjac[d, d, …]`.
#
# Threading: `Threads.@threads :static` over contiguous chunks of the
# flattened `(e fastest, c)` index — the same `divrem(n, nthreads)`
# split as the KA static CPU backend and the NUMA first-touch
# convention of downstream packages. Like KA's static mode, the fast
# path must not be called from inside another threaded region.

# Contiguous balanced chunks of 1:n, one per thread — replicates
# KernelAbstractions' CPU `__run` split (threads 1:rem get len + 1).
function _static_chunks(n::Int, nt::Int = Threads.nthreads())
    nt = clamp(min(nt, n), 1, n)
    len, rem = divrem(n, nt)
    ranges = Vector{UnitRange{Int}}(undef, nt)
    lo = 1
    for t in 1:nt
        hi = lo + len - 1 + (t <= rem ? 1 : 0)
        ranges[t] = lo:hi
        lo = hi + 1
    end
    return ranges
end

# Volume index on the neighbour across face f of element e for the
# face-node (p, q): the same array element `_gather_face3d_batch!`
# would have staged into `ft[pn, qn, nf, nbr, c]` — reading it
# directly is bitwise-identical and skips the gather pass.
@inline function _nbr_volume_index(conn, f::Int, e::Int, p::Int, q::Int,
                                   N::Int)
    nbr = Int(conn.neighbour[f, e])
    nf  = Int(conn.neighbour_face[f, e])
    o   = conn.orientation[f, e]
    pn, qn = _neigh_pq(o, p, q, N)
    an = (nf + 1) >> 1
    rn = isodd(nf) ? 1 : N
    ni, nj, nk = _face_volume_idx(an, rn, pn, qn)
    return nbr, ni, nj, nk
end

# ---- apply_D_batch! ----------------------------------------------------

# Volume part: buf[i,j,k] = (Σ_p G[·,p]·u) · Jd, SIMD across outputs,
# serial in the reduction index p (bitwise: same per-output order as
# the KA kernel's sequential dot).
@inline function _D_volume_elem!(buf, u, Jd, G, e::Int, c::Int,
                                 ::Val{d}, ::Val{N}) where {d, N}
    T = eltype(buf)
    @inbounds if d == 1
        for k in 1:N, j in 1:N
            acc = zero(MVector{N, T})
            for p in 1:N
                up = u[p, j, k, e, c]
                @simd for i in 1:N
                    acc[i] += G[i, p] * up      # G column: contiguous
                end
            end
            @simd for i in 1:N
                buf[i, j, k] = acc[i] * Jd[i, j, k, e]
            end
        end
    elseif d == 2
        for k in 1:N, j in 1:N
            acc = zero(MVector{N, T})
            for p in 1:N
                Gjp = G[j, p]
                @simd for i in 1:N
                    acc[i] += Gjp * u[i, p, k, e, c]
                end
            end
            @simd for i in 1:N
                buf[i, j, k] = acc[i] * Jd[i, j, k, e]
            end
        end
    else
        for k in 1:N
            acc = zero(MMatrix{N, N, T})
            for p in 1:N
                Gkp = G[k, p]
                for j in 1:N
                    @simd for i in 1:N
                        acc[i, j] += Gkp * u[i, j, p, e, c]
                    end
                end
            end
            for j in 1:N
                @simd for i in 1:N
                    buf[i, j, k] = acc[i, j] * Jd[i, j, k, e]
                end
            end
        end
    end
    return nothing
end

# Centred-flux SAT on the two faces normal to axis d (exact expression
# trees of `_apply_D_3d_volume_batch_kernel!`); physical-boundary faces
# (`bdry ≠ 0`) are skipped whole.
@inline function _D_sat_elem!(buf, u, Jd, conn, H11::T, HNN::T,
                              e::Int, c::Int, ::Val{d},
                              ::Val{N}) where {T, d, N}
    half = T(1) / T(2)
    @inbounds for f_idx in 1:2
        f = 2d - 2 + f_idx                  # 2d−1 (low face), 2d (high)
        conn.bdry[f, e] == 0 || continue
        row = f_idx == 1 ? 1 : N
        Hii = f_idx == 1 ? H11 : HNN
        for q in 1:N, p in 1:N
            i, j, k = _face_volume_idx(d, row, p, q)
            nbr, ni, nj, nk = _nbr_volume_index(conn, f, e, p, q, N)
            u_self = u[i, j, k, e, c]
            u_nbr  = u[ni, nj, nk, nbr, c]
            idJ    = Jd[i, j, k, e]
            buf[i, j, k] += f_idx == 1 ?
                (u_self - u_nbr) * half * idJ / Hii :
                (u_nbr - u_self) * half * idJ / Hii
        end
    end
    return nothing
end

# Fused scale / accumulate writeback (the SAT is already in buf, so
# `Du (+)= scale * s` matches the kernel's single store).
@inline function _writeback_elem!(Du, buf, e::Int, c::Int, scale::T,
                                  ::Val{ACC}, ::Val{N}) where {T, ACC, N}
    @inbounds for k in 1:N, j in 1:N
        @simd for i in 1:N
            if ACC
                Du[i, j, k, e, c] += scale * buf[i, j, k]
            else
                Du[i, j, k, e, c] = scale * buf[i, j, k]
            end
        end
    end
    return nothing
end

# `nt` is exposed for the chunking-invariance test only.
function _apply_D_batch_cpu!(Du::AbstractArray{T,5}, u::AbstractArray{T,5},
                             ::Val{d}; geom::MeshGeometry{3, T, N},
                             ops::SBPOps{N, T}, scale::T,
                             accumulate::Val,
                             nt::Int = Threads.nthreads()) where {T, N, d}
    Base.mightalias(Du, u) &&
        throw(ArgumentError("apply_D_batch!: Du must not alias u"))
    Ne = geom.Ne
    C  = size(u, 5)
    Jd   = view(geom.invjacd, :, :, :, :, d)
    conn = geom.conn
    G    = ops.G
    H11  = ops.H[1, 1]
    HNN  = ops.H[N, N]
    ranges = _static_chunks(Ne * C, nt)
    Threads.@threads :static for t in 1:length(ranges)
        buf = Array{T, 3}(undef, N, N, N)
        for ec in ranges[t]
            e = (ec - 1) % Ne + 1               # e fastest: contiguous
            c = (ec - 1) ÷ Ne + 1               # memory slab per thread
            _D_volume_elem!(buf, u, Jd, G, e, c, Val(d), Val(N))
            _D_sat_elem!(buf, u, Jd, conn, H11, HNN, e, c, Val(d), Val(N))
            _writeback_elem!(Du, buf, e, c, scale, accumulate, Val(N))
        end
    end
    return Du
end

# ---- apply_gradient3d_batch! -------------------------------------------

# Conservative split-form gradient, i-pencil layout: SIMD lane = i,
# serial reduction p, at fixed (j, k). The axis-1 operands (up, a1p…)
# are lane-independent scalars; the axis-2/3 operands are contiguous
# lane vectors. Expression trees match `_grad3d_volume_batch_kernel!`
# term for term (bitwise).
@inline function _grad_volume_elem!(bufx, bufy, bufz, u, metric, G,
                                    e::Int, c::Int, ::Val{N}) where {N}
    T = eltype(bufx)
    ax1 = metric.ax1; ax2 = metric.ax2; ax3 = metric.ax3
    ay1 = metric.ay1; ay2 = metric.ay2; ay3 = metric.ay3
    az1 = metric.az1; az2 = metric.az2; az3 = metric.az3
    invdetJ = metric.invdetJ
    half = T(1) / 2
    @inbounds for k in 1:N, j in 1:N
        csx = zero(MVector{N, T}); csy = zero(MVector{N, T})
        csz = zero(MVector{N, T})
        gξ = zero(MVector{N, T}); gη = zero(MVector{N, T})
        gζ = zero(MVector{N, T})
        for p in 1:N
            Gj = G[j, p]; Gk = G[k, p]
            up = u[p, j, k, e, c]
            a1 = ax1[p, j, k, e]; b1 = ay1[p, j, k, e]; c1 = az1[p, j, k, e]
            @simd for i in 1:N
                Gi = G[i, p]
                uq = u[i, p, k, e, c]; ur = u[i, j, p, e, c]
                csx[i] += Gi*a1*up + Gj*ax2[i,p,k,e]*uq + Gk*ax3[i,j,p,e]*ur
                csy[i] += Gi*b1*up + Gj*ay2[i,p,k,e]*uq + Gk*ay3[i,j,p,e]*ur
                csz[i] += Gi*c1*up + Gj*az2[i,p,k,e]*uq + Gk*az3[i,j,p,e]*ur
                gξ[i] += Gi*up; gη[i] += Gj*uq; gζ[i] += Gk*ur
            end
        end
        @simd for i in 1:N
            idJ = invdetJ[i, j, k, e]
            bufx[i,j,k] = half*(csx[i] + ax1[i,j,k,e]*gξ[i] +
                                ax2[i,j,k,e]*gη[i] + ax3[i,j,k,e]*gζ[i])*idJ
            bufy[i,j,k] = half*(csy[i] + ay1[i,j,k,e]*gξ[i] +
                                ay2[i,j,k,e]*gη[i] + ay3[i,j,k,e]*gζ[i])*idJ
            bufz[i,j,k] = half*(csz[i] + az1[i,j,k,e]*gξ[i] +
                                az2[i,j,k,e]*gη[i] + az3[i,j,k,e]*gζ[i])*idJ
        end
    end
    return nothing
end

# Centred-flux SAT for the gradient. Faces run f = 1:6 in order —
# edge/corner nodes sit on 2–3 faces and must accumulate their
# corrections in the same order as the kernel's per-node f loop.
@inline function _grad_sat_elem!(bufx, bufy, bufz, u, metric, conn, H,
                                 e::Int, c::Int, ::Val{N}) where {N}
    T = eltype(bufx)
    invdetJ = metric.invdetJ
    half = T(1) / 2
    @inbounds for f in 1:6
        conn.bdry[f, e] == 0 || continue
        a_idx = (f + 1) >> 1
        row = isodd(f) ? 1 : N
        s_f = isodd(f) ? -one(T) : one(T)
        axA = a_idx == 1 ? metric.ax1 : a_idx == 2 ? metric.ax2 : metric.ax3
        ayA = a_idx == 1 ? metric.ay1 : a_idx == 2 ? metric.ay2 : metric.ay3
        azA = a_idx == 1 ? metric.az1 : a_idx == 2 ? metric.az2 : metric.az3
        Hrr = H[row, row]
        for q in 1:N, p in 1:N
            i, jj, k = _face_volume_idx(a_idx, row, p, q)
            nbr, ni, nj, nk = _nbr_volume_index(conn, f, e, p, q, N)
            jump = u[ni, nj, nk, nbr, c] - u[i, jj, k, e, c]
            idJ = invdetJ[i, jj, k, e]
            cf = idJ * half * jump / Hrr
            bufx[i, jj, k] += cf * s_f * axA[i, jj, k, e]
            bufy[i, jj, k] += cf * s_f * ayA[i, jj, k, e]
            bufz[i, jj, k] += cf * s_f * azA[i, jj, k, e]
        end
    end
    return nothing
end

function _apply_gradient3d_batch_cpu!(g1::AbstractArray{T,5},
                                      g2::AbstractArray{T,5},
                                      g3::AbstractArray{T,5},
                                      u::AbstractArray{T,5};
                                      geom::MeshGeometry{3, T, N},
                                      ops::SBPOps{N, T}, metric,
                                      nt::Int = Threads.nthreads()) where {T, N}
    (Base.mightalias(g1, u) || Base.mightalias(g2, u) ||
     Base.mightalias(g3, u)) &&
        throw(ArgumentError("apply_gradient3d_batch!: outputs must not alias u"))
    Ne = geom.Ne
    C  = size(u, 5)
    conn = geom.conn
    G = ops.G
    H = ops.H
    ranges = _static_chunks(Ne * C, nt)
    Threads.@threads :static for t in 1:length(ranges)
        bufx = Array{T, 3}(undef, N, N, N)
        bufy = Array{T, 3}(undef, N, N, N)
        bufz = Array{T, 3}(undef, N, N, N)
        for ec in ranges[t]
            e = (ec - 1) % Ne + 1
            c = (ec - 1) ÷ Ne + 1
            _grad_volume_elem!(bufx, bufy, bufz, u, metric, G, e, c, Val(N))
            _grad_sat_elem!(bufx, bufy, bufz, u, metric, conn, H, e, c, Val(N))
            @inbounds for k in 1:N, j in 1:N
                @simd for i in 1:N
                    g1[i, j, k, e, c] = bufx[i, j, k]
                    g2[i, j, k, e, c] = bufy[i, j, k]
                    g3[i, j, k, e, c] = bufz[i, j, k]
                end
            end
        end
    end
    return g1, g2, g3
end

# ---- apply_divergence3d_batch! -----------------------------------------

# Divergence volume part. The KA kernel keeps 10 scalar accumulators in
# one p loop; here the p loop is split into one pass per accumulator
# group (cs; then gξ/gη/gζ per flux component) to bound register
# pressure — bitwise-safe because every accumulator still sees its own
# complete p sequence in order.
@inline function _div_volume_elem!(bufd, F1, F2, F3, metric, G,
                                   e::Int, c::Int, ::Val{N}) where {N}
    T = eltype(bufd)
    ax1 = metric.ax1; ax2 = metric.ax2; ax3 = metric.ax3
    ay1 = metric.ay1; ay2 = metric.ay2; ay3 = metric.ay3
    az1 = metric.az1; az2 = metric.az2; az3 = metric.az3
    invdetJ = metric.invdetJ
    half = T(1) / 2
    @inbounds for k in 1:N, j in 1:N
        cs = zero(MVector{N, T})
        for p in 1:N
            Gj = G[j, p]; Gk = G[k, p]
            # Axis-1 contravariant flux: lane-independent scalar.
            Fξ = ax1[p,j,k,e]*F1[p,j,k,e,c] + ay1[p,j,k,e]*F2[p,j,k,e,c] +
                 az1[p,j,k,e]*F3[p,j,k,e,c]
            @simd for i in 1:N
                Gi = G[i, p]
                Fη = ax2[i,p,k,e]*F1[i,p,k,e,c] + ay2[i,p,k,e]*F2[i,p,k,e,c] +
                     az2[i,p,k,e]*F3[i,p,k,e,c]
                Fζ = ax3[i,j,p,e]*F1[i,j,p,e,c] + ay3[i,j,p,e]*F2[i,j,p,e,c] +
                     az3[i,j,p,e]*F3[i,j,p,e,c]
                cs[i] += Gi*Fξ + Gj*Fη + Gk*Fζ
            end
        end
        gξ1 = zero(MVector{N, T}); gη1 = zero(MVector{N, T})
        gζ1 = zero(MVector{N, T})
        for p in 1:N
            Gj = G[j, p]; Gk = G[k, p]
            Fp = F1[p, j, k, e, c]
            @simd for i in 1:N
                gξ1[i] += G[i, p]*Fp
                gη1[i] += Gj*F1[i, p, k, e, c]
                gζ1[i] += Gk*F1[i, j, p, e, c]
            end
        end
        gξ2 = zero(MVector{N, T}); gη2 = zero(MVector{N, T})
        gζ2 = zero(MVector{N, T})
        for p in 1:N
            Gj = G[j, p]; Gk = G[k, p]
            Fp = F2[p, j, k, e, c]
            @simd for i in 1:N
                gξ2[i] += G[i, p]*Fp
                gη2[i] += Gj*F2[i, p, k, e, c]
                gζ2[i] += Gk*F2[i, j, p, e, c]
            end
        end
        gξ3 = zero(MVector{N, T}); gη3 = zero(MVector{N, T})
        gζ3 = zero(MVector{N, T})
        for p in 1:N
            Gj = G[j, p]; Gk = G[k, p]
            Fp = F3[p, j, k, e, c]
            @simd for i in 1:N
                gξ3[i] += G[i, p]*Fp
                gη3[i] += Gj*F3[i, p, k, e, c]
                gζ3[i] += Gk*F3[i, j, p, e, c]
            end
        end
        @simd for i in 1:N
            ad = ax1[i,j,k,e]*gξ1[i] + ax2[i,j,k,e]*gη1[i] + ax3[i,j,k,e]*gζ1[i] +
                 ay1[i,j,k,e]*gξ2[i] + ay2[i,j,k,e]*gη2[i] + ay3[i,j,k,e]*gζ2[i] +
                 az1[i,j,k,e]*gξ3[i] + az2[i,j,k,e]*gη3[i] + az3[i,j,k,e]*gζ3[i]
            bufd[i, j, k] = half * (cs[i] + ad) * invdetJ[i, j, k, e]
        end
    end
    return nothing
end

@inline function _div_sat_elem!(bufd, F1, F2, F3, metric, conn, H,
                                e::Int, c::Int, ::Val{N}) where {N}
    T = eltype(bufd)
    invdetJ = metric.invdetJ
    half = T(1) / 2
    @inbounds for f in 1:6
        conn.bdry[f, e] == 0 || continue
        a_idx = (f + 1) >> 1
        row = isodd(f) ? 1 : N
        s_f = isodd(f) ? -one(T) : one(T)
        axA = a_idx == 1 ? metric.ax1 : a_idx == 2 ? metric.ax2 : metric.ax3
        ayA = a_idx == 1 ? metric.ay1 : a_idx == 2 ? metric.ay2 : metric.ay3
        azA = a_idx == 1 ? metric.az1 : a_idx == 2 ? metric.az2 : metric.az3
        Hrr = H[row, row]
        for q in 1:N, p in 1:N
            i, jj, k = _face_volume_idx(a_idx, row, p, q)
            nbr, ni, nj, nk = _nbr_volume_index(conn, f, e, p, q, N)
            axc = axA[i, jj, k, e]; ayc = ayA[i, jj, k, e]
            azc = azA[i, jj, k, e]
            Fn_self = s_f * (axc*F1[i,jj,k,e,c] + ayc*F2[i,jj,k,e,c] +
                             azc*F3[i,jj,k,e,c])
            Fn_nbr  = s_f * (axc*F1[ni,nj,nk,nbr,c] +
                             ayc*F2[ni,nj,nk,nbr,c] +
                             azc*F3[ni,nj,nk,nbr,c])
            idJ = invdetJ[i, jj, k, e]
            bufd[i, jj, k] += idJ * half * (Fn_nbr - Fn_self) / Hrr
        end
    end
    return nothing
end

function _apply_divergence3d_batch_cpu!(divF::AbstractArray{T,5},
                                        F1::AbstractArray{T,5},
                                        F2::AbstractArray{T,5},
                                        F3::AbstractArray{T,5};
                                        geom::MeshGeometry{3, T, N},
                                        ops::SBPOps{N, T}, metric,
                                        add::Union{Nothing,AbstractArray{T,5}},
                                        nt::Int = Threads.nthreads()) where {T, N}
    (Base.mightalias(divF, F1) || Base.mightalias(divF, F2) ||
     Base.mightalias(divF, F3)) &&
        throw(ArgumentError("apply_divergence3d_batch!: divF must not alias the fluxes"))
    Ne = geom.Ne
    C  = size(F1, 5)
    conn = geom.conn
    G = ops.G
    H = ops.H
    ranges = _static_chunks(Ne * C, nt)
    Threads.@threads :static for t in 1:length(ranges)
        bufd = Array{T, 3}(undef, N, N, N)
        for ec in ranges[t]
            e = (ec - 1) % Ne + 1
            c = (ec - 1) ÷ Ne + 1
            _div_volume_elem!(bufd, F1, F2, F3, metric, G, e, c, Val(N))
            _div_sat_elem!(bufd, F1, F2, F3, metric, conn, H, e, c, Val(N))
            if add === nothing
                @inbounds for k in 1:N, j in 1:N
                    @simd for i in 1:N
                        divF[i, j, k, e, c] = bufd[i, j, k]
                    end
                end
            else
                @inbounds for k in 1:N, j in 1:N
                    @simd for i in 1:N
                        divF[i, j, k, e, c] = bufd[i, j, k] +
                                              add[i, j, k, e, c]
                    end
                end
            end
        end
    end
    return divF
end
