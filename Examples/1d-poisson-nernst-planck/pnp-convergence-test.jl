using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature, Cubature, Cuba
using QuasiMonteCarlo

include("1d-poisson-nerst-planck-Cl-Na-adim-neuralpde.jl")

# Define strategies ############################################################

grid_strategy = NeuralPDE.GridTraining(dx=1e-2)
stochastic_strategy = NeuralPDE.StochasticTraining(number_of_points=100)
quasirandom_strategy = NeuralPDE.QuasiRandomTraining(sampling_method = UniformSample(),
                                                     number_of_points = 100,
                                                     number_of_minibatch = 100)
strategies = [grid_strategy, stochastic_strategy, quasirandom_strategy]
qalgs = [HCubatureJL(), CubatureJLh(), CubatureJLp()]


# Solve PNP using different strategies #########################################

losses = []
reses = []
for strategy in strategies
    res, loss, discretization, pars = solve_PNP(strategy)
    push!(losses, loss)
    push!(reses, res)
end

qlosses = []
qreses = []
for alg in qalgs
    strategy = QuadratureTraining(algorithm =alg,reltol=1e-5,
                                  abstol=1e-5, maxiters=50)
    res, loss, discretization, pars = solve_PNP(strategy)
    push!(qreses, res)
    push!(qlosses, loss)
end

# Plot and save results ########################################################

using Plots

plot(1:length(losses[1]), losses[1], yscale = :log10,
    xlabel="No. of training steps", ylabel="Loss", label="Grid strategy")
plot!(1:length(losses[2]), losses[2], yscale = :log10,
    xlabel="No. of training steps",ylabel="Loss",label="Stochastic strategy")
plot!(1:length(losses[3]), losses[3], yscale = :log10,
    xlabel="No. of training steps", ylabel="Loss",label="Quasirandom strategy")

plot!(1:length(qlosses[1]), qlosses[1], yscale = :log10,
    xlabel="No. of training steps", ylabel="Loss", label="HCubatureJL")
plot!(1:length(qlosses[2]), qlosses[2], yscale = :log10,
    xlabel="No. of training steps",ylabel="Loss",label="CubatureJLh")
plot!(1:length(qlosses[3]), qlosses[3], yscale = :log10,
    xlabel="No. of training steps", ylabel="Loss", label="CubatureJLp")

savefig("pnp-convergence-test.svg")



