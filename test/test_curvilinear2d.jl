# Tests for the free-stream-preserving conservative first-derivative on
# curvilinear 2D meshes: make_metric_terms2d + apply_gradient2d! /
# apply_divergence2d! (split skew-symmetric form). Cubed-square mesh.

using HexMeshes: make_cubed_square_mesh
using HexSBPSAT
using LinearAlgebra
using Test

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan);
                       flush(stderr)))

@testset "curvilinear 2D gradient/divergence" begin
    T = Float64; N = 4

    _progress("free-stream (∇const = 0, ∇·const = 0)")
    @testset "free-stream preservation" begin
        for M in (2, 3)
            mesh = make_cubed_square_mesh(T, M, T(0.3))
            elem = make_element(T, N); ops = make_operators(elem)
            geom = make_geometry(mesh, elem); metric = make_metric_terms2d(geom, ops)
            Ne = geom.Ne
            g1 = zeros(T, N, N, Ne); g2 = similar(g1)
            apply_gradient2d!(g1, g2, fill(T(2.5), N, N, Ne); geom, ops, metric)
            @test maximum(abs, g1) ≤ 1e-10
            @test maximum(abs, g2) ≤ 1e-10
            dv = similar(g1)
            apply_divergence2d!(dv, fill(T(1.3), N, N, Ne), fill(T(-0.7), N, N, Ne);
                                geom, ops, metric)
            @test maximum(abs, dv) ≤ 1e-10
        end
    end

    _progress("interior skew-adjointness (gradient = −divergence*)")
    @testset "interior gradient/divergence skew-adjoint" begin
        M = 2
        mesh = make_cubed_square_mesh(T, M, T(0.3))
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem); metric = make_metric_terms2d(geom, ops)
        Ne = geom.Ne; n = N*N*Ne
        Gx = zeros(T, n, n); Dx = zeros(T, n, n)
        u = zeros(T, N, N, Ne); t1 = similar(u); t2 = similar(u); z = zeros(T, N, N, Ne)
        for jc in 1:n
            fill!(u, 0); u[jc] = 1
            apply_gradient2d!(t1, t2, u; geom, ops, metric); Gx[:, jc] = vec(t1)
            apply_divergence2d!(t1, u, z; geom, ops, metric); Dx[:, jc] = vec(t1)
        end
        Hd = Diagonal(vec(metric.Hd))
        M_mis = Hd*Gx + (Hd*Dx)'
        # Zero the outer-boundary face nodes; the interior must be
        # exactly skew-adjoint (the boundary term is handled by BCs).
        bnode = falses(N, N, Ne)
        for e in 1:Ne, f in 1:4
            geom.conn.bdry[f, e] == 0 && continue
            row = isodd(f) ? 1 : N
            for p in 1:N
                f ≤ 2 ? (bnode[row, p, e] = true) : (bnode[p, row, e] = true)
            end
        end
        bidx = findall(vec(bnode))
        M_int = copy(M_mis); M_int[bidx, :] .= 0; M_int[:, bidx] .= 0
        @test norm(M_int) ≤ 1e-10 * norm(Hd*Gx)
    end

    _progress("consistency / convergence")
    @testset "gradient converges on a smooth field" begin
        f(x, y)  = sin(x) * cos(y)
        fx(x, y) =  cos(x) * cos(y)
        fy(x, y) = -sin(x) * sin(y)
        errs = T[]
        for M in (2, 4, 8)
            mesh = make_cubed_square_mesh(T, M, T(0.3))
            elem = make_element(T, N); ops = make_operators(elem)
            geom = make_geometry(mesh, elem); metric = make_metric_terms2d(geom, ops)
            X = geom.coords[1, :, :, :]; Y = geom.coords[2, :, :, :]
            g1 = similar(X); g2 = similar(X)
            apply_gradient2d!(g1, g2, f.(X, Y); geom, ops, metric)
            e1 = g1 .- fx.(X, Y); e2 = g2 .- fy.(X, Y)
            push!(errs, sqrt(sum(@. (e1^2 + e2^2) * metric.Hd)))
        end
        @test all(isfinite, errs)
        @test errs[end] < errs[1]
        @test (errs[1] / errs[end])^(1 / (length(errs) - 1)) > 1.5
    end
end
