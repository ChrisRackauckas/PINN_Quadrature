using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature,Cubature,Cuba
using Plots

# Physical and numerical parameters (fixed) ####################################
nu = 0.07
nx = 10001 #101
x_max = 2.0 * pi
dx = x_max / (nx - 1.0)
nt = 2 #10
dt = dx * nu 
t_max = dt * nt

# Analytic function ############################################################
@parameters x t
analytic_sol_func(x, t) = -2*nu*(-(-8*t + 2*x)*exp(-(-4*t + x)^2/(4*nu*(t + 1)))/
                          (4*nu*(t + 1)) - (-8*t + 2*x - 12.5663706143592)*
                          exp(-(-4*t + x - 6.28318530717959)^2/(4*nu*(t + 1)))/
                          (4*nu*(t + 1)))/(exp(-(-4*t + x - 6.28318530717959)^2/
                          (4*nu*(t + 1))) + exp(-(-4*t + x)^2/(4*nu*(t + 1)))) + 4
domains = [x ∈ IntervalDomain(0.0, x_max),
           t ∈ IntervalDomain(0.0, t_max)]
xs, ts = [domain.domain.lower:dx:domain.domain.upper for (dx, domain) in zip([dx, dt], domains)]
u_real = reshape([analytic_sol_func(x, t) for t in ts for x in xs], (length(xs), length(ts)))

function solve(opt)
    strategy = QuadratureTraining()

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eq = Dt(u(x, t)) + u(x, t) * Dx(u(x, t))  ~ nu * Dxx(u(x, t)) 

    bcs = [u(x,0.0) ~ analytic_sol_func(x, 0.0),
           u(0.0,t) ~ u(x_max,t)]

    domains = [x ∈ IntervalDomain(0.0, x_max),
               t ∈ IntervalDomain(0.0, t_max)]

    discretization = PhysicsInformedNN(chain, strategy)

    indvars = [x, t]   #physically independent variables
    depvars = [u]       #dependent (target) variable

    loss = []
    initial_time = 0

    times = []

    cb = function (p,l)
        if initial_time == 0
            initial_time = time()
        end
        push!(times, time() - initial_time)
        println("Current loss for $opt is: $l")
        push!(loss, l)
        return false
    end

    pde_system = PDESystem(eq, bcs, domains, indvars, depvars)
    prob = discretize(pde_system, discretization)

    if opt == "both"
        res = GalacticOptim.solve(prob, ADAM(); cb = cb, maxiters=50)
        prob = remake(prob,u0=res.minimizer)
        res = GalacticOptim.solve(prob, BFGS(); cb = cb, maxiters=150)
    else
        res = GalacticOptim.solve(prob, opt; cb = cb, maxiters=200)
    end

    times[1] = 0.001
    
    return loss, times #add numeric solution
end

opt1 = ADAM()
opt2 = ADAM(0.005)
opt3 = ADAM(0.05)
opt4 = RMSProp()
opt5 = RMSProp(0.005)
opt6 = RMSProp(0.05)
opt7 = Optim.BFGS()
opt8 = Optim.LBFGS()

loss_1, times_1 = solve(opt1)
loss_2, times_2 = solve(opt2)
loss_3, times_3 = solve(opt3)
loss_4, times_4 = solve(opt4)
loss_5, times_5 = solve(opt5)
loss_6, times_6 = solve(opt6)
loss_7, times_7 = solve(opt7)
loss_8, times_8 = solve(opt8)
loss_9, times_9 = solve("both")

p = plot([times_1, times_2, times_3, times_4, times_5, times_6, times_7, times_8, times_9], [loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, loss_8, loss_9],xlabel="time (s)", ylabel="loss", xscale=:log, yscale=:log, labels=["ADAM(0.001)" "ADAM(0.005)" "ADAM(0.05)" "RMSProp(0.001)" "RMSProp(0.005)" "RMSProp(0.05)" "BFGS()" "LBFGS()" "ADAM + BFGS"], legend=:bottomleft, linecolor=["#2660A4" "#4CD0F4" "#FEC32F" "#F763CD" "#44BD79" "#831894" "#A6ED18" "#980000" "#FF912B"])
savefig("burgers_equation_loss_over_time.png")
p = plot(1:201, [loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, loss_8, loss_9[2:end]], xlabel="iterations", ylabel="loss", yscale=:log, labels=["ADAM(0.001)" "ADAM(0.005)" "ADAM(0.05)" "RMSProp(0.001)" "RMSProp(0.005)" "RMSProp(0.05)" "BFGS()" "LBFGS()" "ADAM + BFGS"], legend=:bottomleft, linecolor=["#2660A4" "#4CD0F4" "#FEC32F" "#F763CD" "#44BD79" "#831894" "#A6ED18" "#980000" "#FF912B"])
savefig("burgers_equation_loss_over_iterations.png")

@show loss_1[end], loss_2[end], loss_3[end], loss_4[end], loss_5[end], loss_6[end], loss_7[end], loss_8[end], loss_9[end]