# Channel-batched 3D operators ≡ the scalar operators applied per
# channel, to round-off, on affine and curvilinear meshes; plus the
# fused scale/accumulate/add variants.

using HexMeshes: make_uniform_hex, make_cubed_cube_mesh,
                 make_radial_shell_mesh
using HexSBPSAT
using Random
using Test

@testset "kernels3d_batch" begin
    T = Float64
    N = 4

    function batch_setup(mesh; nchannels)
        elem = make_element(T, N)
        ops = make_operators(elem)
        geom = make_geometry(mesh, elem)
        work = make_workspace(geom; nchannels)
        work_s = make_workspace(geom)
        return elem, ops, geom, work, work_s
    end

    meshes = [
        ("uniform periodic", make_uniform_hex(T, 2, 2, 2, T(0), T(1);
                                              periodic = true), false),
        ("uniform dirichlet", make_uniform_hex(T, 2, 2, 2, T(0), T(1);
                                               periodic = false), false),
        ("cubed cube", make_cubed_cube_mesh(T, 2, T(0.4)), true),
        ("radial shell", make_radial_shell_mesh(T, T(1.0), T(2.0), 2;
                                                M_r = 2), true),
    ]

    @testset "$name, C=$C" for (name, mesh, curv) in meshes,
                               C in (1, 3, 10)
        elem, ops, geom, work, work_s = batch_setup(mesh; nchannels = 3C)
        Ne = geom.Ne
        rng = Random.MersenneTwister(7 + C)
        u = rand(rng, T, N, N, N, Ne, C)

        if !curv
            # apply_D_batch! ≡ apply_D! per channel, all axes.
            Du = fill(T(NaN), N, N, N, Ne, C)
            Du_ref = similar(Du)
            for d in 1:3
                apply_D_batch!(Du, u, d; geom, ops, work)
                for c in 1:C
                    apply_D!(view(Du_ref, :, :, :, :, c),
                             view(u, :, :, :, :, c), d;
                             geom, ops, work = work_s)
                end
                @test maximum(abs, Du - Du_ref) < 1e-13
            end

            # scale and accumulate variants.
            apply_D_batch!(Du, u, 1; geom, ops, work)
            acc = copy(Du)
            apply_D_batch!(acc, u, 2; geom, ops, work, scale = T(0.5),
                           accumulate = Val(true))
            Du2 = similar(Du)
            apply_D_batch!(Du2, u, 2; geom, ops, work)
            @test maximum(abs, acc - (Du + T(0.5) * Du2)) < 1e-13
        else
            metric = make_metric_terms3d(geom, ops)

            # Batched gradient ≡ scalar gradient per channel.
            g1 = fill(T(NaN), N, N, N, Ne, C); g2 = similar(g1); g3 = similar(g1)
            apply_gradient3d_batch!(g1, g2, g3, u; geom, ops, metric, work)
            r1 = similar(g1); r2 = similar(g1); r3 = similar(g1)
            for c in 1:C
                apply_gradient3d!(view(r1, :, :, :, :, c),
                                  view(r2, :, :, :, :, c),
                                  view(r3, :, :, :, :, c),
                                  view(u, :, :, :, :, c);
                                  geom, ops, metric, work = work_s)
            end
            @test maximum(abs, g1 - r1) < 1e-13
            @test maximum(abs, g2 - r2) < 1e-13
            @test maximum(abs, g3 - r3) < 1e-13

            # Batched divergence ≡ scalar divergence per channel; fused add.
            F1 = rand(rng, T, N, N, N, Ne, C)
            F2 = rand(rng, T, N, N, N, Ne, C)
            F3 = rand(rng, T, N, N, N, Ne, C)
            dv = fill(T(NaN), N, N, N, Ne, C)
            apply_divergence3d_batch!(dv, F1, F2, F3; geom, ops, metric, work)
            dr = similar(dv)
            for c in 1:C
                apply_divergence3d!(view(dr, :, :, :, :, c),
                                    view(F1, :, :, :, :, c),
                                    view(F2, :, :, :, :, c),
                                    view(F3, :, :, :, :, c);
                                    geom, ops, metric, work = work_s)
            end
            @test maximum(abs, dv - dr) < 1e-13

            add = rand(rng, T, N, N, N, Ne, C)
            dva = similar(dv)
            apply_divergence3d_batch!(dva, F1, F2, F3; geom, ops, metric,
                                      work, add)
            @test maximum(abs, dva - (dr + add)) < 1e-13
        end
    end
end
