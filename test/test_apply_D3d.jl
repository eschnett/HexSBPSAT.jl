# Tests for the 3D axis-selective first-derivative operator
# `apply_D!(Du, u, d; geom::MeshGeometry{3}, ops, work)` (reference SBP-G
# along axis d + centred-flux SAT). Axis-aligned affine hex meshes.

using HexMeshes: make_uniform_hex
using HexSBPSAT
using LinearAlgebra
using Test

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan);
                       flush(stderr)))

function _dense_D3d(geom, ops, d, ::Type{T}, N, Ne) where {T}
    n = N * N * N * Ne
    A = zeros(T, n, n)
    work = make_workspace(geom)
    u = zeros(T, N, N, N, Ne); Du = similar(u)
    for j in 1:n
        fill!(u, 0); u[j] = 1
        apply_D!(Du, u, d; geom, ops, work)
        A[:, j] = vec(Du)
    end
    return A
end

@testset "apply_D! 3D" begin
    T = Float64

    _progress("H·D skewness (periodic, all axes)")
    @testset "H·D exactly skew on periodic hex" begin
        for N in (4,), M in (2,)
            mesh = make_uniform_hex(T, M, 0.0, 1.0; periodic = true)
            elem = make_element(T, N); ops = make_operators(elem)
            geom = make_geometry(mesh, elem)
            Hdiag = Diagonal(vec(geom.Hphys))
            for d in (1, 2, 3)
                A = _dense_D3d(geom, ops, d, T, N, geom.Ne)
                HD = Hdiag * A
                @test norm(HD + HD') ≤ 1000 * eps(T) * norm(HD)
            end
        end
    end

    _progress("directional correctness + convergence")
    @testset "∂x / ∂y / ∂z on smooth periodic data" begin
        N = 4
        ex = Float64[]
        for M in (4, 8, 16)
            mesh = make_uniform_hex(T, M, 0.0, 1.0; periodic = true)
            elem = make_element(T, N); ops = make_operators(elem)
            geom = make_geometry(mesh, elem); work = make_workspace(geom)
            xs = geom.coords[1, :, :, :, :]
            zs = geom.coords[3, :, :, :, :]
            ux = sinpi.(2 .* xs); Dux = similar(ux)
            apply_D!(Dux, ux, 1; geom, ops, work)
            push!(ex, maximum(abs.(Dux .- 2π .* cospi.(2 .* xs))))
            # ∂x of a z-only field must vanish.
            uz = sinpi.(2 .* zs); Duxz = similar(uz)
            apply_D!(Duxz, uz, 1; geom, ops, work)
            @test maximum(abs.(Duxz)) ≤ 1e-10
            # ∂z of a z-field is correct.
            Duz = similar(uz); apply_D!(Duz, uz, 3; geom, ops, work)
            @test maximum(abs.(Duz .- 2π .* cospi.(2 .* zs))) < 1
        end
        @test all(>(2.4), log2.(ex[1:end-1] ./ ex[2:end]))
    end

    _progress("polynomial exactness (non-periodic interior)")
    @testset "polynomial exactness" begin
        N, M = 5, 2
        mesh = make_uniform_hex(T, M, -1.0, 1.0; periodic = false)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem); work = make_workspace(geom)
        xs = geom.coords[1, :, :, :, :]; ys = geom.coords[2, :, :, :, :]
        for p in 0:N-1
            u = xs .^ p; Du = similar(u); apply_D!(Du, u, 1; geom, ops, work)
            exact = p == 0 ? zero(xs) : p .* xs .^ (p - 1)
            @test maximum(abs.(Du .- exact)) ≤ 1e-9 * max(1, M^p)
            u2 = ys .^ p; Du2 = similar(u2); apply_D!(Du2, u2, 2; geom, ops, work)
            exact2 = p == 0 ? zero(ys) : p .* ys .^ (p - 1)
            @test maximum(abs.(Du2 .- exact2)) ≤ 1e-9 * max(1, M^p)
        end
    end

    _progress("device round-trip (CPU)")
    @testset "to_device CPU round-trip" begin
        using KernelAbstractions: CPU
        N, M = 4, 3
        mesh = make_uniform_hex(T, M, 0.0, 1.0; periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem); work = make_workspace(geom)
        geom2 = to_device(geom, CPU()); work2 = to_device(work, CPU())
        u = rand(T, N, N, N, geom.Ne)
        for d in (1, 2, 3)
            Du = similar(u); Du2 = similar(u)
            apply_D!(Du, u, d; geom, ops, work)
            apply_D!(Du2, u, d; geom = geom2, ops, work = work2)
            @test Du == Du2
        end
    end
end

# GPU smoke test (Metal/CUDA): apply_D! 3D on-device matches CPU.
if !@isdefined(_HAS_GPU_3D)
    const _HAS_GPU_3D, _GPU_BACKEND_3D = try
        if Sys.isapple() && Sys.ARCH === :aarch64
            @eval using Metal
            Metal.functional() ? (true, Metal.MetalBackend()) : (false, nothing)
        elseif !Sys.isapple()
            @eval using CUDA
            CUDA.functional() ? (true, CUDA.CUDABackend()) : (false, nothing)
        else
            (false, nothing)
        end
    catch
        (false, nothing)
    end
end

if _HAS_GPU_3D
    @testset "apply_D! 3D on GPU (Float32)" begin
        _progress("GPU vs CPU agreement (Float32)")
        using KernelAbstractions
        Tg = Float32; N = 4; M = 3
        mesh = make_uniform_hex(Tg, M, 0.0f0, 1.0f0; periodic = true)
        elem = make_element(Tg, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem); work = make_workspace(geom)
        u = rand(Tg, N, N, N, geom.Ne)
        geom_d = to_device(geom, _GPU_BACKEND_3D)
        work_d = to_device(work, _GPU_BACKEND_3D)
        u_d = KernelAbstractions.allocate(_GPU_BACKEND_3D, Tg, N, N, N, geom.Ne)
        copyto!(u_d, u)
        for d in (1, 2, 3)
            Du = similar(u); apply_D!(Du, u, d; geom, ops, work)
            Du_d = similar(u_d); apply_D!(Du_d, u_d, d; geom = geom_d, ops, work = work_d)
            @test Array(Du_d) ≈ Du rtol = 1e-5
        end
    end
end
