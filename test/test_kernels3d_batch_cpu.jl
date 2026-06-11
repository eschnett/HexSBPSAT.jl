# The plain-Julia CPU fast path of the channel-batched operators must
# agree BITWISE with the KernelAbstractions implementation (`@test ==`,
# not `≈`): both evaluate every output with the identical per-output
# reduction order; the fast path's SIMD runs across outputs only, and
# its direct neighbour reads are verbatim the values the gather pass
# would have staged. Any divergence here means one of the two paths
# changed its expression tree.

using HexSBPSAT
using HexSBPSAT: _apply_D_batch_ka!, _apply_D_batch_cpu!, _static_chunks
using HexMeshes
using KernelAbstractions
using Random
using Test

@testset "static chunking" begin
    @test _static_chunks(10, 4) == [1:3, 4:6, 7:8, 9:10]
    @test _static_chunks(4, 8) == [1:1, 2:2, 3:3, 4:4]
    @test _static_chunks(7, 1) == [1:7]
    for (n, nt) in ((100, 7), (3, 3), (64, 64))
        rs = _static_chunks(n, nt)
        @test first(rs[1]) == 1 && last(rs[end]) == n
        @test all(first(rs[i+1]) == last(rs[i]) + 1 for i in 1:length(rs)-1)
    end
end

@testset "apply_D_batch! CPU fast path ≡ KA (bitwise)" begin
    Random.seed!(42)
    meshes(T) = [
        ("uniform periodic", make_uniform_hex(T, 3, 3, 3, T(0), T(1);
                                              periodic = true)),
        ("uniform dirichlet", make_uniform_hex(T, 3, 3, 3, T(0), T(1);
                                               periodic = false)),
        # cubed cube exercises nontrivial face orientations
        ("cubed cube", make_cubed_cube_mesh(T, 2, T(0.4))),
    ]
    @testset "T = $T" for T in (Float64, Float32)
        N = 4
        @testset "$name" for (name, mesh) in meshes(T)
            elem = make_element(T, N)
            ops = make_operators(elem)
            geom = make_geometry(mesh, elem)
            mw = make_workspace(geom; nchannels = 20)
            Ne = geom.Ne
            for C in (1, 10, 20), d in 1:3
                u = rand(T, N, N, N, Ne, C)
                # plain
                Dka = fill(T(NaN), N, N, N, Ne, C)
                Dfp = fill(T(NaN), N, N, N, Ne, C)
                _apply_D_batch_ka!(Dka, u, d; geom, ops, work = mw,
                                   backend = CPU())
                apply_D_batch!(Dfp, u, d; geom, ops, work = mw,
                               backend = CPU(static = true))
                @test Dka == Dfp
                # scale + accumulate into a prefilled output
                D0 = rand(T, N, N, N, Ne, C)
                Dka2 = copy(D0); Dfp2 = copy(D0)
                _apply_D_batch_ka!(Dka2, u, d; geom, ops, work = mw,
                                   scale = T(3)/T(2),
                                   accumulate = Val(true), backend = CPU())
                apply_D_batch!(Dfp2, u, d; geom, ops, work = mw,
                               scale = T(3)/T(2), accumulate = Val(true),
                               backend = CPU(static = true))
                @test Dka2 == Dfp2
                # contiguous-view inputs (the GH call pattern)
                if C >= 10
                    uv = view(u, :, :, :, :, 1:10)
                    Dv1 = view(Dka, :, :, :, :, 1:10)
                    Dv2 = view(Dfp, :, :, :, :, 1:10)
                    _apply_D_batch_ka!(Dv1, uv, d; geom, ops, work = mw,
                                       backend = CPU())
                    apply_D_batch!(Dv2, uv, d; geom, ops, work = mw)
                    @test Dv1 == Dv2
                end
            end
        end
    end

    @testset "chunking invariance" begin
        T = Float64; N = 4
        mesh = make_uniform_hex(T, 2, 2, 2, T(0), T(1); periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem)
        u = rand(T, N, N, N, geom.Ne, 5)
        ref = similar(u); out = similar(u)
        _apply_D_batch_cpu!(ref, u, Val(2); geom, ops, scale = one(T),
                            accumulate = Val(false), nt = 1)
        for nt in (2, 3, Threads.nthreads())
            _apply_D_batch_cpu!(out, u, Val(2); geom, ops, scale = one(T),
                                accumulate = Val(false), nt = nt)
            @test out == ref
        end
    end

    @testset "aliasing guard" begin
        T = Float64; N = 4
        mesh = make_uniform_hex(T, 1, 1, 1, T(0), T(1); periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem)
        mw = make_workspace(geom; nchannels = 2)
        u = rand(T, N, N, N, 1, 2)
        @test_throws ArgumentError apply_D_batch!(u, u, 1; geom, ops,
                                                  work = mw)
    end
end

@testset "gradient/divergence CPU fast path ≡ KA (bitwise)" begin
    Random.seed!(7)
    using HexSBPSAT: _apply_gradient3d_batch_ka!, _apply_divergence3d_batch_ka!,
                     make_metric_terms3d
    curv_meshes(T) = [
        ("radial shell", make_radial_shell_mesh(T, T(1), T(2), 2; M_r = 2)),
        ("inflated cube", make_inflated_cube_mesh(T, T(0.2), T(0.5), T(1),
                                                  2; outer_bc = :dirichlet)),
        ("cubed cube", make_cubed_cube_mesh(T, 2, T(0.4))),
        ("warped periodic", make_warped_uniform_hex(T, 2, T(0), T(1),
                                                    T(0.05);
                                                    periodic = true)),
    ]
    @testset "T = $T" for T in (Float64, Float32)
        N = 4
        @testset "$name" for (name, mesh) in curv_meshes(T)
            elem = make_element(T, N)
            ops = make_operators(elem)
            geom = make_geometry(mesh, elem)
            metric = make_metric_terms3d(geom, ops)
            Ne = geom.Ne
            for C in (1, 10)
                mw = make_workspace(geom; nchannels = 3C)
                u = rand(T, N, N, N, Ne, C)
                # gradient
                gs_ka = ntuple(_ -> fill(T(NaN), N, N, N, Ne, C), 3)
                gs_fp = ntuple(_ -> fill(T(NaN), N, N, N, Ne, C), 3)
                _apply_gradient3d_batch_ka!(gs_ka..., u; geom, ops, metric,
                                            work = mw, backend = CPU())
                apply_gradient3d_batch!(gs_fp..., u; geom, ops, metric,
                                        work = mw)
                @test gs_ka[1] == gs_fp[1]
                @test gs_ka[2] == gs_fp[2]
                @test gs_ka[3] == gs_fp[3]
                # divergence, with and without fused add
                F1, F2, F3 = (rand(T, N, N, N, Ne, C) for _ in 1:3)
                dka = fill(T(NaN), N, N, N, Ne, C)
                dfp = fill(T(NaN), N, N, N, Ne, C)
                _apply_divergence3d_batch_ka!(dka, F1, F2, F3; geom, ops,
                                              metric, work = mw,
                                              backend = CPU())
                apply_divergence3d_batch!(dfp, F1, F2, F3; geom, ops,
                                          metric, work = mw)
                @test dka == dfp
                src = rand(T, N, N, N, Ne, C)
                _apply_divergence3d_batch_ka!(dka, F1, F2, F3; geom, ops,
                                              metric, work = mw, add = src,
                                              backend = CPU())
                apply_divergence3d_batch!(dfp, F1, F2, F3; geom, ops,
                                          metric, work = mw, add = src)
                @test dka == dfp
            end
        end
    end
end
