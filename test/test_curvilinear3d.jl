# Tests for the free-stream-preserving conservative first-derivative on
# curvilinear 3D meshes: make_metric_terms3d (conservative-curl form) +
# apply_gradient3d! / apply_divergence3d!. cubed-cube / inflated-cube /
# radial-shell.

using HexMeshes: make_cubed_cube_mesh, make_inflated_cube_mesh,
                 make_radial_shell_mesh
using HexSBPSAT
using LinearAlgebra
using Test

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan);
                       flush(stderr)))

_curv3d_meshes(::Type{T}) where {T} = (
    ("cubed_cube",    make_cubed_cube_mesh(T, 2, T(0.3))),
    ("inflated_cube", make_inflated_cube_mesh(T, T(0.2), T(0.5), T(1.0), 2)),
    ("radial_shell",  make_radial_shell_mesh(T, T(0.5), T(1.0), 2)),
)

@testset "curvilinear 3D gradient/divergence" begin
    T = Float64; N = 4

    _progress("metric identities (GCL) ≤ round-off")
    @testset "conservative-curl GCL [$name]" for (name, mesh) in _curv3d_meshes(T)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem); metric = make_metric_terms3d(geom, ops)
        G = ops.G; Ne = geom.Ne
        a = (metric.ax1,metric.ax2,metric.ax3, metric.ay1,metric.ay2,metric.ay3,
             metric.az1,metric.az2,metric.az3)
        worst = 0.0
        for e in 1:Ne, n in 0:2
            aξ=a[3n+1]; aη=a[3n+2]; aζ=a[3n+3]
            for k in 1:N, j in 1:N, i in 1:N
                d = 0.0
                for p in 1:N
                    d += G[i,p]*aξ[p,j,k,e] + G[j,p]*aη[i,p,k,e] + G[k,p]*aζ[i,j,p,e]
                end
                worst = max(worst, abs(d))
            end
        end
        @test worst ≤ 1e-10
    end

    _progress("free-stream (∇const = 0, ∇·const = 0)")
    @testset "free-stream [$name]" for (name, mesh) in _curv3d_meshes(T)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem); metric = make_metric_terms3d(geom, ops)
        work = make_workspace(geom); Ne = geom.Ne
        g1=zeros(T,N,N,N,Ne); g2=similar(g1); g3=similar(g1)
        apply_gradient3d!(g1,g2,g3, fill(T(2.5),N,N,N,Ne); geom,ops,metric,work)
        @test maximum(abs,g1) ≤ 1e-10 && maximum(abs,g2) ≤ 1e-10 && maximum(abs,g3) ≤ 1e-10
        dv=similar(g1)
        apply_divergence3d!(dv, fill(T(1.3),N,N,N,Ne), fill(T(-0.7),N,N,N,Ne),
                            fill(T(0.4),N,N,N,Ne); geom,ops,metric,work)
        @test maximum(abs,dv) ≤ 1e-10
    end

    _progress("interior skew-adjointness (gradient = −divergence*)")
    @testset "skew-adjoint (radial_shell M=1)" begin
        mesh = make_radial_shell_mesh(T, T(0.5), T(1.0), 1)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem); metric = make_metric_terms3d(geom, ops)
        work = make_workspace(geom); Ne = geom.Ne; n = N^3*Ne
        Gx = zeros(T,n,n); Dx = zeros(T,n,n)
        u=zeros(T,N,N,N,Ne); t1=similar(u);t2=similar(u);t3=similar(u); z=zeros(T,N,N,N,Ne)
        for jc in 1:n
            fill!(u,0); u[jc]=1
            apply_gradient3d!(t1,t2,t3,u; geom,ops,metric,work); Gx[:,jc]=vec(t1)
            apply_divergence3d!(t1,u,z,z; geom,ops,metric,work); Dx[:,jc]=vec(t1)
        end
        Hd = Diagonal(vec(metric.Hd)); Mmis = Hd*Gx + (Hd*Dx)'
        bnode = falses(N,N,N,Ne)
        for e in 1:Ne, f in 1:6
            geom.conn.bdry[f,e]==0 && continue
            a_idx=(f+1)÷2; row=isodd(f) ? 1 : N
            for q in 1:N, p in 1:N
                a_idx==1 ? (bnode[row,p,q,e]=true) :
                a_idx==2 ? (bnode[p,row,q,e]=true) : (bnode[p,q,row,e]=true)
            end
        end
        bidx=findall(vec(bnode)); Mint=copy(Mmis); Mint[bidx,:].=0; Mint[:,bidx].=0
        @test norm(Mint) ≤ 1e-10 * norm(Hd*Gx)
    end

    _progress("consistency / convergence")
    @testset "gradient converges on a smooth field" begin
        f(x,y,z)  =  sin(x)*cos(y)*sin(z)
        fx(x,y,z) =  cos(x)*cos(y)*sin(z)
        errs = T[]
        for M in (2, 4)
            mesh = make_cubed_cube_mesh(T, M, T(0.3))
            elem = make_element(T, N); ops = make_operators(elem)
            geom = make_geometry(mesh, elem); metric = make_metric_terms3d(geom, ops)
            work = make_workspace(geom)
            X=geom.coords[1,:,:,:,:]; Y=geom.coords[2,:,:,:,:]; Z=geom.coords[3,:,:,:,:]
            g1=similar(X);g2=similar(X);g3=similar(X)
            apply_gradient3d!(g1,g2,g3, f.(X,Y,Z); geom,ops,metric,work)
            push!(errs, sqrt(sum(@. (g1 - fx(X,Y,Z))^2 * metric.Hd)))
        end
        @test all(isfinite, errs)
        @test errs[1]/errs[2] > 2
    end
end

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
    @testset "curvilinear 3D on GPU (Float32)" begin
        _progress("GPU vs CPU curvilinear 3D operator agreement (Float32)")
        using KernelAbstractions
        Tg = Float32; N = 4
        mk(a) = (d = KernelAbstractions.allocate(_GPU_BACKEND_3D, Tg, size(a));
                 copyto!(d, a); d)
        for (name, mesh) in _curv3d_meshes(Tg)
            elem = make_element(Tg, N); ops = make_operators(elem)
            geom = make_geometry(mesh, elem); metric = make_metric_terms3d(geom, ops)
            work = make_workspace(geom); Ne = geom.Ne
            Φ = rand(Tg,N,N,N,Ne); F1=rand(Tg,N,N,N,Ne); F2=rand(Tg,N,N,N,Ne); F3=rand(Tg,N,N,N,Ne)
            g1=similar(Φ);g2=similar(Φ);g3=similar(Φ); dv=similar(Φ)
            apply_gradient3d!(g1,g2,g3,Φ; geom,ops,metric,work)
            apply_divergence3d!(dv,F1,F2,F3; geom,ops,metric,work)
            gd=to_device(geom,_GPU_BACKEND_3D); wd=to_device(work,_GPU_BACKEND_3D)
            md=metric_to_device(metric,_GPU_BACKEND_3D)
            Φd=mk(Φ);F1d=mk(F1);F2d=mk(F2);F3d=mk(F3)
            g1d=similar(Φd);g2d=similar(Φd);g3d=similar(Φd);dvd=similar(Φd)
            apply_gradient3d!(g1d,g2d,g3d,Φd; geom=gd,ops,metric=md,work=wd)
            apply_divergence3d!(dvd,F1d,F2d,F3d; geom=gd,ops,metric=md,work=wd)
            @test Array(g1d) ≈ g1 rtol=1e-4 atol=1e-4
            @test Array(dvd) ≈ dv rtol=1e-4 atol=1e-4
        end
    end
end
