# Tests for `apply_laplacian!` — the curvilinear-aware 2D SBP-DG
# Laplacian — and its diagnostics (`discrete_laplacian`,
# `spectral_radius_estimate`, `discrete_inner_product`,
# `discrete_l2_norm`). Mirrors the structural checks in
# `test_apply_laplacian3d.jl` adapted to two dimensions.

using HexSBPSAT
using HexSBPSAT: make_element, make_operators, make_geometry,
                 make_workspace,
                 apply_laplacian!, discrete_laplacian,
                 spectral_radius_estimate, discrete_inner_product,
                 discrete_l2_norm, physical_mass_diagonal, to_device
using HexMeshes: make_uniform_quad
using KernelAbstractions: CPU
using LinearAlgebra
using Random
using Test

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan);
                       flush(stderr)))

@testset "apply_laplacian! 2D (T=$T)" for T in (Float64, Float32)

    sym_tol = T === Float64 ? 1.0e-9 : 1.0e-4
    nsd_tol = T === Float64 ? 1.0e-6 : 1.0e-2
    sr_rtol = T === Float64 ? 0.05  : 0.10
    ip_rtol = T === Float64 ? 1.0e-12 : 1.0e-5

    _progress("explicit 2D Laplacian assembly: symmetric, NSD (T=$T)")
    @testset "discrete_laplacian: mass-symmetric and NSD (T=$T)" begin
        N    = 3
        M    = 2
        elem = make_element(T, N)
        ops  = make_operators(elem)
        mesh = make_uniform_quad(T, M, zero(T), one(T))
        geom = make_geometry(mesh, elem)
        τ    = T(3//2) * (N - 1)^2

        L  = Matrix(discrete_laplacian(geom, ops, τ))
        Hd = physical_mass_diagonal(geom, ops)
        S  = Diagonal(Hd) * L

        @test maximum(abs.(S - S')) < sym_tol      # mass-symmetric
        λs = eigvals(Symmetric((S + S') / 2))
        @test maximum(λs) < nsd_tol
    end

    _progress("spectral_radius_estimate matches max|λ| of assembled L (T=$T)")
    @testset "spectral_radius_estimate against full spectrum (T=$T)" begin
        N    = 3
        M    = 2
        elem = make_element(T, N)
        ops  = make_operators(elem)
        mesh = make_uniform_quad(T, M, zero(T), one(T))
        geom = make_geometry(mesh, elem)
        τ    = T(3//2) * (N - 1)^2

        L = Matrix(discrete_laplacian(geom, ops, τ))
        λ_dense = maximum(abs, eigvals(L))
        Random.seed!(20260528)
        λ_iter  = spectral_radius_estimate(geom, ops, τ)
        @test isapprox(λ_iter, λ_dense; rtol = sr_rtol)
    end

    _progress("tag-7 free face contributes nothing — but is reachable (T=$T)")
    @testset "tag-7 face is a free face (T=$T)" begin
        # Inject a tag ≥ 5 on every outer face and verify that
        # apply_laplacian!'s output:
        #   1. differs from the Dirichlet-default version
        #   2. stays finite (clean no-op, not NaN-producing skip).
        N    = 3
        M    = 2
        elem = make_element(T, N)
        ops  = make_operators(elem)
        mesh = make_uniform_quad(T, M, zero(T), one(T))
        geom = make_geometry(mesh, elem)
        work = make_workspace(geom)
        τ    = T(3//2) * (N - 1)^2

        u   = randn(T, N, N, mesh.Ne)
        ü_d = similar(u)
        apply_laplacian!(ü_d, u, ntuple(_ -> zero(T), Val(4)); geom, ops, work, τ)

        for e in 1:mesh.Ne, f in 1:4
            if geom.conn.neighbour[f, e] == 0
                geom.conn.bdry[f, e] = Int8(7)
            end
        end
        ü_s = similar(u)
        apply_laplacian!(ü_s, u, ntuple(_ -> zero(T), Val(4)); geom, ops, work, τ)

        @test ü_d != ü_s
        @test all(isfinite, ü_s)
    end

    _progress("free face implements natural lift: L_h(x²) = 2 (T=$T)")
    @testset "free face: natural lift recovers L_h(x²) = 2 (T=$T)" begin
        # Mirror of the 3D test. On an axis-aligned quad mesh the SBP
        # operator is exact for polynomials of degree ≤ N-1, so
        # L_h(x²) must equal ∇²(x²) = 2 at every node including outer-
        # face nodes — verifying the natural boundary lift is present
        # on tag-7 free faces.
        N    = 3
        M    = 4
        elem = make_element(T, N)
        ops  = make_operators(elem)
        mesh = make_uniform_quad(T, M, -one(T), one(T))
        for e in 1:mesh.Ne, f in 1:4
            if mesh.conn.neighbour[f, e] == 0
                mesh.conn.bdry[f, e] = Int8(7)
            end
        end
        geom = make_geometry(mesh, elem)
        work = make_workspace(geom)
        u = T.(geom.coords[1, :, :, :] .^ 2)
        ü = similar(u)
        τ = T(3//2) * (N - 1)^2
        apply_laplacian!(ü, u, ntuple(_ -> zero(T), Val(4)); geom, ops, work, τ)
        @test maximum(abs, ü .- T(2)) < 100 * eps(T)
    end

    _progress("inner product / L² norm consistency (T=$T)")
    @testset "discrete_inner_product / l2_norm 2D (T=$T)" begin
        N    = 3
        M    = 2
        elem = make_element(T, N)
        ops  = make_operators(elem)
        mesh = make_uniform_quad(T, M, zero(T), one(T))
        geom = make_geometry(mesh, elem)

        u = ones(T, N, N, mesh.Ne)
        v = ones(T, N, N, mesh.Ne)
        ip = discrete_inner_product(u, v, geom, ops)
        @test isapprox(ip, sum(geom.Hphys); rtol = ip_rtol)
        @test isapprox(ip, one(T); rtol = ip_rtol)   # area of unit square

        nrm = discrete_l2_norm(u, geom, ops)
        @test isapprox(nrm, sqrt(ip); rtol = ip_rtol)
    end

    _progress("to_device CPU round-trip (T=$T)")
    @testset "to_device round-trip on CPU backend (T=$T)" begin
        N    = 3
        M    = 2
        elem = make_element(T, N)
        ops  = make_operators(elem)
        mesh = make_uniform_quad(T, M, zero(T), one(T))
        geom = make_geometry(mesh, elem)
        τ    = T(3//2) * (N - 1)^2

        geom_dev = to_device(geom, CPU())
        @test geom_dev.coords ≈ geom.coords
        @test geom_dev !== geom

        work     = make_workspace(geom)
        work_dev = to_device(work, CPU())

        u  = randn(T, N, N, mesh.Ne)
        bdry = ntuple(_ -> zero(T), Val(4))
        ü_host = similar(u); ü_dev = similar(u)
        apply_laplacian!(ü_host, u, bdry; geom,            ops, work,             τ)
        apply_laplacian!(ü_dev,  u, bdry; geom = geom_dev, ops, work = work_dev, τ)
        @test ü_host == ü_dev
    end

end
