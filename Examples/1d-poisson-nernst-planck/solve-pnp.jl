include("1d-poisson-nerst-planck-Cl-Na-adim-neuralpde.jl")
res, loss, discretization,pars = solve_PNP()
plot_PNP(res, loss, discretization, pars)
