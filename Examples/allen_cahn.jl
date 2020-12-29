using NeuralPDE
using Quadrature, Cubature, Cuba
using Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Plots
using PyPlot
using DelimitedFiles
using QuasiMonteCarlo

using DifferentialEquations
using LinearAlgebra

print("Precompiling Done")

d = 20 # number of dimensions
X0 = fill(0.0f0, d) # initial value of stochastic control process
tspan = (0.3f0,0.6f0)
dt = 0.015 # time step
m = 100 # number of trajectories (batch size)


g(X) = 1/(2.0f0 + 0.4f0 * sum(X.^2))
μ_f(X,p,t) = zero(X)  # Vector d x 1 λ
σ_f(X,p,t) = Diagonal(sqrt(2.0f0) * ones(Float32, d)) # Matrix d x d
f(X,u,σᵀ∇u,p,t) = u .- u.^3


prob = TerminalPDEProblem(g, f, μ_f, σ_f, X0, tspan)

hls = 10 + d # hidden layer size
opt = Flux.ADAM(0.001)  # optimizer
# sub-neural network approximating solutions at the desired point
u0 = Flux.Chain(Dense(d, hls, relu),
                Dense(hls, hls, relu),
                Dense(hls, 1))

σᵀ∇u = Flux.Chain(Dense(d + 1, hls, relu),
                  Dense(hls, hls, relu),
                  Dense(hls, hls, relu),
                  Dense(hls, d))

pdealg = NNPDENS(u0, σᵀ∇u, opt=opt)

@time ans = solve(prob, pdealg, verbose=true, maxiters=200, trajectories=m,
                            alg=EM(), dt=dt, pabstol=0.01)
