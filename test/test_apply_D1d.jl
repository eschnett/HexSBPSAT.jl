# Tests for the 1D MeshGeometry path (`make_geometry(::Mesh{1})`) and
# the connectivity-driven first-derivative operator `apply_D!`
# (SBP-G + centred-flux SAT). Operator-level identities only; wave
# evolution tests live in WaveToySecondOrder.

using HexMeshes: make_uniform_line
using HexSBPSAT
using LinearAlgebra
using Test

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan);
                       flush(stderr)))

# Assemble the dense matrix of `apply_D!` by column probing.
function _dense_D(geom, ops, ::Type{T}, N, M) where {T}
    n = N * M
    A = zeros(T, n, n)
    u = zeros(T, N, M)
    Du = similar(u)
    for j in 1:n
        fill!(u, 0); u[j] = 1
        apply_D!(Du, u; geom, ops)
        A[:, j] = vec(Du)
    end
    return A
end

@testset "apply_D! 1D" begin
    T = Float64

    _progress("MeshGeometry{1} construction")
    @testset "make_geometry(::Mesh{1})" begin
        N, M = 4, 8
        mesh = make_uniform_line(T, M, 0.0, 2.0; periodic = true)
        elem = make_element(T, N)
        ops  = make_operators(elem)
        geom = make_geometry(mesh, elem)
        h = 2.0 / M

        @test geom.Ne == M
        @test size(geom.coords) == (1, N, M)
        @test size(geom.jac)    == (1, 1, N, M)
        @test size(geom.invjac) == (1, 1, N, M)
        @test size(geom.detjac) == (N, M)
        @test size(geom.Hphys)  == (N, M)
        @test all(geom.handedness .== 1)
        # Affine elements: J = h everywhere, Hphys = H_ref · h.
        @test all(geom.jac .≈ h)
        @test all(geom.invjac .≈ 1 / h)
        @test all(geom.dinvjac .== 0)
        for i in 1:N
            @test all(geom.Hphys[i, :] .≈ ops.H[i, i] * h)
        end
        # Collocation coordinates: element m covers [x0 + (m-1)h, x0 + mh].
        @test geom.coords[1, 1, 1] ≈ 0.0
        @test geom.coords[1, N, M] ≈ 2.0
        # element_coords wrapper agrees.
        @test element_coords(mesh, elem) == geom.coords
    end

    _progress("H·D skewness (periodic)")
    @testset "H·D exactly skew on periodic mesh" begin
        for N in (4, 8), M in (1, 4)
            mesh = make_uniform_line(T, M, 0.0, 1.0; periodic = true)
            elem = make_element(T, N)
            ops  = make_operators(elem)
            geom = make_geometry(mesh, elem)
            A = _dense_D(geom, ops, T, N, M)
            HD = Diagonal(vec(geom.Hphys)) * A
            @test norm(HD + HD') ≤ 100 * eps(T) * norm(HD)
        end
    end

    _progress("polynomial exactness (non-periodic interior)")
    @testset "polynomial exactness" begin
        # On a non-periodic line, polynomial data of degree ≤ N−1 is
        # continuous across all interior faces (SAT vanishes) and the
        # outer faces carry no SAT, so the derivative is exact at every
        # node.
        N, M = 5, 4
        mesh = make_uniform_line(T, M, -1.0, 1.0; periodic = false)
        elem = make_element(T, N)
        ops  = make_operators(elem)
        geom = make_geometry(mesh, elem)
        xs = geom.coords[1, :, :]
        for p in 0:N-1
            u  = xs .^ p
            Du = similar(u)
            apply_D!(Du, u; geom, ops)
            exact = p == 0 ? zero(xs) : p .* xs .^ (p - 1)
            @test maximum(abs.(Du .- exact)) ≤ 1000 * eps(T) * max(1, M^p)
        end
    end

    _progress("periodic neighbour wiring")
    @testset "connectivity matches mod1 wiring" begin
        # On the uniform periodic line the connectivity-driven SAT must
        # reproduce the hardwired mod1(m ± 1, M) neighbour relation.
        N, M = 4, 8
        mesh = make_uniform_line(T, M, 0.0, 1.0; periodic = true)
        elem = make_element(T, N)
        ops  = make_operators(elem)
        geom = make_geometry(mesh, elem)
        h = 1.0 / M

        u  = rand(T, N, M)
        Du = similar(u)
        apply_D!(Du, u; geom, ops)

        ref = similar(u)
        G = Matrix(ops.G)
        for m in 1:M
            ref[:, m] = (G * u[:, m]) ./ h
        end
        c1 = 1 / (2 * ops.H[1, 1] * h)
        cN = 1 / (2 * ops.H[N, N] * h)
        for m in 1:M
            mL = mod1(m - 1, M)
            mR = mod1(m + 1, M)
            ref[1, m] += c1 * (u[1, m] - u[N, mL])
            ref[N, m] += cN * (u[1, mR] - u[N, m])
        end
        @test Du ≈ ref rtol = 100 * eps(T)
    end

    _progress("spectral convergence on smooth periodic data")
    @testset "convergence on sin(2πx)" begin
        N = 4
        errs = T[]
        for M in (8, 16, 32)
            mesh = make_uniform_line(T, M, 0.0, 1.0; periodic = true)
            elem = make_element(T, N)
            ops  = make_operators(elem)
            geom = make_geometry(mesh, elem)
            xs = geom.coords[1, :, :]
            u  = sinpi.(2 .* xs)
            Du = similar(u)
            apply_D!(Du, u; geom, ops)
            exact = 2 * T(π) .* cospi.(2 .* xs)
            push!(errs, maximum(abs.(Du .- exact)))
        end
        rates = log2.(errs[1:end-1] ./ errs[2:end])
        @test all(rates .> N - 1.5)   # ≈ N−1 for the derivative + SAT
    end

    _progress("device round-trip (CPU)")
    @testset "to_device CPU round-trip" begin
        using KernelAbstractions: CPU
        N, M = 4, 8
        mesh = make_uniform_line(T, M, 0.0, 1.0; periodic = true)
        elem = make_element(T, N)
        ops  = make_operators(elem)
        geom = make_geometry(mesh, elem)

        mesh2 = to_device(mesh, CPU())
        @test mesh2.conn.neighbour == mesh.conn.neighbour
        geom2 = to_device(geom, CPU())
        @test geom2.coords == geom.coords
        @test geom2.Hphys == geom.Hphys
        work = make_workspace(geom)
        @test size(work.face_trace) == (2, 2, M)

        u  = rand(T, N, M)
        Du = similar(u); Du2 = similar(u)
        apply_D!(Du, u; geom, ops)
        apply_D!(Du2, u; geom = geom2, ops)
        @test Du == Du2
    end
end
