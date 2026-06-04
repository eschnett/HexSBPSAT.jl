# Tests for the 2D axis-selective first-derivative operator
# `apply_D!(Du, u, d; geom::MeshGeometry{2}, ops)` (reference SBP-G
# along axis d + centred-flux SAT). Axis-aligned affine meshes.

using HexMeshes: make_uniform_quad
using HexSBPSAT
using LinearAlgebra
using Test

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan);
                       flush(stderr)))

function _dense_D2d(geom, ops, d, ::Type{T}, N, Ne) where {T}
    n = N * N * Ne
    A = zeros(T, n, n)
    u = zeros(T, N, N, Ne); Du = similar(u)
    for j in 1:n
        fill!(u, 0); u[j] = 1
        apply_D!(Du, u, d; geom, ops)
        A[:, j] = vec(Du)
    end
    return A
end

@testset "apply_D! 2D" begin
    T = Float64

    _progress("H·D skewness (periodic, both axes)")
    @testset "H·D exactly skew on periodic quad" begin
        for N in (4, 6), M in (2, 4)
            mesh = make_uniform_quad(T, M, M, 0.0, 1.0; periodic = true)
            elem = make_element(T, N); ops = make_operators(elem)
            geom = make_geometry(mesh, elem)
            Hdiag = Diagonal(vec(geom.Hphys))
            for d in (1, 2)
                A = _dense_D2d(geom, ops, d, T, N, geom.Ne)
                HD = Hdiag * A
                @test norm(HD + HD') ≤ 1000 * eps(T) * norm(HD)
            end
        end
    end

    _progress("directional correctness + convergence")
    @testset "∂x / ∂y on smooth periodic data" begin
        N = 4
        ex, ey = Float64[], Float64[]
        for M in (4, 8, 16)
            mesh = make_uniform_quad(T, M, M, 0.0, 1.0; periodic = true)
            elem = make_element(T, N); ops = make_operators(elem)
            geom = make_geometry(mesh, elem)
            xs = geom.coords[1, :, :, :]; ys = geom.coords[2, :, :, :]

            ux = sinpi.(2 .* xs); Dux = similar(ux)
            apply_D!(Dux, ux, 1; geom, ops)
            push!(ex, maximum(abs.(Dux .- 2π .* cospi.(2 .* xs))))
            # ∂x of a y-only field must vanish.
            uy = sinpi.(2 .* ys); Duxy = similar(uy)
            apply_D!(Duxy, uy, 1; geom, ops)
            @test maximum(abs.(Duxy)) ≤ 1e-10

            Duy = similar(uy)
            apply_D!(Duy, uy, 2; geom, ops)
            push!(ey, maximum(abs.(Duy .- 2π .* cospi.(2 .* ys))))
        end
        @test all(>(2.4), log2.(ex[1:end-1] ./ ex[2:end]))
        @test all(>(2.4), log2.(ey[1:end-1] ./ ey[2:end]))
    end

    _progress("polynomial exactness (non-periodic interior)")
    @testset "polynomial exactness" begin
        # Degree ≤ N−1 polynomials are continuous across interior faces
        # (SAT vanishes); outer faces carry no SAT → exact at all nodes.
        N, M = 5, 3
        mesh = make_uniform_quad(T, M, M, -1.0, 1.0; periodic = false)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem)
        xs = geom.coords[1, :, :, :]; ys = geom.coords[2, :, :, :]
        for p in 0:N-1
            u = xs .^ p; Du = similar(u); apply_D!(Du, u, 1; geom, ops)
            exact = p == 0 ? zero(xs) : p .* xs .^ (p - 1)
            @test maximum(abs.(Du .- exact)) ≤ 1e-9 * max(1, M^p)
            u2 = ys .^ p; Du2 = similar(u2); apply_D!(Du2, u2, 2; geom, ops)
            exact2 = p == 0 ? zero(ys) : p .* ys .^ (p - 1)
            @test maximum(abs.(Du2 .- exact2)) ≤ 1e-9 * max(1, M^p)
        end
    end
end
