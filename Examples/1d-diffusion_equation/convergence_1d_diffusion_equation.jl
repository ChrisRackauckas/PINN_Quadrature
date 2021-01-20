using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature,Cubature, Cuba
using QuasiMonteCarlo


function convergence_1d_diffusion_equation(strategy)
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

    discretization = NeuralPDE.PhysicsInformedNN(chain,
                                                 strategy = strategy)
    phi = discretization.phi
    pde_system = PDESystem(eq,bcs,domains,[x,t],[u])
    prob = NeuralPDE.discretize(pde_system,discretization)

    loss_list = []
    cb = function (p,l)
        push!(loss_list, l)
        return false
    end

    res = GalacticOptim.solve(prob, ADAM(0.001); cb=cb, maxiters=5000)

    return loss_list,res
end


losses = []
reses = []
grid_strategy = NeuralPDE.GridTraining(dx=[0.2,0.1])
stochastic_strategy = NeuralPDE.StochasticTraining(number_of_points=100)
quasirandom_strategy = NeuralPDE.QuasiRandomTraining(sampling_method = UniformSample(),
                                                     number_of_points = 100,
                                                     number_of_minibatch = 100)

strategies = [grid_strategy,stochastic_strategy,quasirandom_strategy]

for strategy in strategies
    loss,res = convergence_1d_diffusion_equation(strategy)
    push!(losses,loss)
    push!(reses,res)
end

qalgs = [HCubatureJL(), CubatureJLh(), CubatureJLp()]
qlosses = []
qreses = []
for alg in qalgs
    strategy = QuadratureTraining(algorithm =alg,reltol=1e-5,abstol=1e-5,maxiters=50)
    loss,res = convergence_1d_diffusion_equation(strategy)
    push!(qreses,res)
    push!(qlosses,loss)
end

using Plots
s = 1:length(losses[1])

plot(s,losses[1],yscale = :log10, label = "grid strategy")
plot!(s,losses[2],yscale = :log10,label = "stochastic strategy")
plot!(s,losses[3],yscale = :log10,label = "quasirandom strategy")

plot!(s,qlosses[1],yscale = :log10,label = "HCubatureJL")
plot!(s,qlosses[2],yscale = :log10,label = "CubatureJLh")
plot!(s,qlosses[3],yscale = :log10,label = "CubatureJLp")

savefig("convergence")
