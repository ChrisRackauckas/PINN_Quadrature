using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature,Cubature, Cuba
using QuasiMonteCarlo
using Statistics


cb = function (p,l)
    println("Current loss is: $l")
    return false
end
function solve_1d_diffusion_equation(strategy)
    @parameters x t
    @variables u(..)
    @derivatives Dt'~t
    @derivatives Dxx''~x

    eq = Dt(u(x,t)) - Dxx(u(x,t)) ~ -exp(-t) * (sin(pi * x) - pi^2 * sin(pi * x))

    bcs = [u(x,0) ~ sin(pi*x),
           u(-1,t) ~ 0.,
           u(1,t) ~ 0.]

    domains = [x ∈ IntervalDomain(-1.0,1.0),
               t ∈ IntervalDomain(0.0,1.0)]

    chain = FastChain(FastDense(2,18,Flux.σ),FastDense(18,18,Flux.σ),FastDense(18,1))

    discretization = PhysicsInformedNN(chain,strategy =strategy)

    pde_system = PDESystem(eq,bcs,domains,[x,t],[u])
    prob = discretize(pde_system,discretization)

    res = GalacticOptim.solve(prob, ADAM(0.01); cb = cb, maxiters=5000)
    phi = discretization.phi
    res,phi
end

reses = []
losses = []
phis = []

grid_strategy = GridTraining(dx=[0.2,0.1])
stochastic_strategy = StochasticTraining(number_of_points=100)
quasirandom_strategy = QuasiRandomTraining(sampling_method = UniformSample(),
                                                     number_of_points = 100,
                                                     number_of_minibatch = 100)
quadrature_strategy = QuadratureTraining(algorithm = CubatureJLp(),reltol=1e-5,abstol=1e-5,maxiters=50)

strategies = [grid_strategy,stochastic_strategy,quasirandom_strategy,quadrature_strategy]

for strategy in strategies
    res,phi = solve_1d_diffusion_equation(strategy)
    push!(phis,phi)
    push!(reses,res)
end

@parameters x t
domains = [x ∈ IntervalDomain(-1.0,1.0),
           t ∈ IntervalDomain(0.0,1.0)]
dx = 0.2; dt = 0.1
xs,ts = [domain.domain.lower:dx/10:domain.domain.upper for (dx,domain) in zip([dx,dt],domains)]
analytic_sol_func(x,t) =  sin(pi*x) * exp(-t)
u_real = reshape([analytic_sol_func(x,t) for x in xs for t in ts], (length(xs),length(ts)))

u_predicts = []
diff_us = []
total_errors = []
for (phi,res) in zip(phis, reses)
    u_predict = reshape([first(phi([x,t],res.minimizer)) for x in xs for t in ts],(length(xs),length(ts)))
    diff_u = abs.(u_predict .- u_real)
    push!(total_errors , mean(diff_u))
    push!(u_predicts,u_predict)
    push!(diff_us,diff_u)
end

using Plots
names  = ["grid","stochastic", "quasirandom", "quadrature" ]
for (u_predict, diff_u,strategy,name) in zip(u_predicts,diff_us,strategies,names)
    p1 = plot(xs, ts, u_real, linetype=:contourf,title = "analytic");
    p2 = plot(xs, ts, u_predict, linetype=:contourf,title = "predict $name");
    p3 = plot(xs, ts, diff_u,linetype=:contourf,title = "error");
    plot(p1,p2,p3)
    savefig("$name")
end

println(total_errors)
"""
grid : 0.002867885455279286
stochastic : 0.17766174285388261
quasirandom : 0.010827789369252872
quadrature : 0.001244297165375137
"""
