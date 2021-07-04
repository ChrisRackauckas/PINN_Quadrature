using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature,Cubature,Cuba
using Plots

t_ref = 1.0       # s
x_ref = 0.38      # dm 
C_ref = 0.16      # mol/dm^3
Phi_ref = 1.0     # V

epsilon = 78.5    # K
F = 96485.3415    # A s mol^-1
R = 831.0         # kg dm^2 s^-2 K^-1 mol^-1 
T = 298.0         # K

z_Na = 1.0        # non-dim
z_Cl = -1.0       # non-dim

D_Na = 0.89e-7    # dm^2 s^−1
D_Cl = 1.36e-7    # dm^2 s^−1

u_Na = D_Na * abs(z_Na) * F / (R * T)
u_Cl = D_Cl * abs(z_Cl) * F / (R * T)

t_max = 0.01 / t_ref    # non-dim
x_max = 0.38 / x_ref    # non-dim
Na_0 = 0.16 / C_ref     # non-dim
Cl_0 = 0.16 / C_ref     # non-dim
Phi_0 = 4.0 / Phi_ref   # non-dim

Na_anode = 0.0            # non-dim
Na_cathode = 2.0 * Na_0   # non-dim
Cl_anode = 1.37 * Cl_0    # non-dim
Cl_cathode = 0.0          # non-dim

Pe_Na = x_ref^2 / ( t_ref * D_Na )  # non-dim
Pe_Cl = x_ref^2 / ( t_ref * D_Cl )  # non-dim

M_Na = x_ref^2 / ( t_ref * Phi_ref * u_Na )  # non-dim
M_Cl = x_ref^2 / ( t_ref * Phi_ref * u_Cl )  # non-dim

Po_1 = (epsilon * Phi_ref) / (F * x_ref * C_ref)  # non-dim

dx = 0.01 # non-dim

function solve(opt)
    strategy = QuadratureTraining()

    @parameters t,x
    @variables Phi(..),Na(..),Cl(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eqs = [
            ( Dxx(Phi(t,x)) ~ ( 1.0 / Po_1 ) *
                              ( z_Na * Na(t,x) + z_Cl * Cl(t,x) ) )
            ,
            ( Dt(Na(t,x)) ~ ( 1.0 / Pe_Na ) * Dxx(Na(t,x)) 
                          +   z_Na / ( abs(z_Na) * M_Na ) 
                          * ( Dx(Na(t,x)) * Dx(Phi(t,x)) + Na(t,x) * Dxx(Phi(t,x)) ) )
            ,
            ( Dt(Cl(t,x)) ~ ( 1.0 / Pe_Cl ) * Dxx(Cl(t,x)) 
                          +   z_Cl / ( abs(z_Cl) * M_Cl ) 
                          * ( Dx(Cl(t,x)) * Dx(Phi(t,x)) + Cl(t,x) * Dxx(Phi(t,x)) ) )
          ]

    bcs = [
            Phi(t,0.0) ~ Phi_0,
            Phi(t,x_max) ~ 0.0
            ,
            Na(0.0,x) ~ Na_0,
            Na(t,0.0) ~ Na_anode,
            Na(t,x_max) ~ Na_cathode
            ,
            Cl(0.0,x) ~ Cl_0,
            Cl(t,0.0) ~ Cl_anode,
            Cl(t,x_max) ~ Cl_cathode
          ]

    # Space and time domains ###################################################

    domains = [
                t ∈ IntervalDomain(0.0, t_max),
                x ∈ IntervalDomain(0.0, x_max)
              ]

    # Neural network, Discretization ###########################################

    dim = length(domains)
    output = length(eqs)
    neurons = 16
    chain1 = FastChain( FastDense(dim, neurons, Flux.σ),
                        FastDense(neurons, neurons, Flux.σ),
                        FastDense(neurons, neurons, Flux.σ),
                        FastDense(neurons, 1))
    chain2 = FastChain( FastDense(dim, neurons, Flux.σ),
                        FastDense(neurons, neurons, Flux.σ),
                        FastDense(neurons, neurons, Flux.σ),
                        FastDense(neurons, 1))
    chain3 = FastChain( FastDense(dim, neurons, Flux.σ),
                        FastDense(neurons, neurons, Flux.σ),
                        FastDense(neurons, neurons, Flux.σ),
                        FastDense(neurons, 1))

    discretization = PhysicsInformedNN([chain1, chain2, chain3], strategy)

    indvars = [t, x]   #phisically independent variables
    depvars = [Phi, Na, Cl]       #dependent (target) variable

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

    pde_system = PDESystem(eqs, bcs, domains, indvars, depvars)
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
savefig("pnp_loss_over_time.png")
p = plot(1:201, [loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, loss_8, loss_9[2:end]], xlabel="iterations", ylabel="loss", yscale=:log, labels=["ADAM(0.001)" "ADAM(0.005)" "ADAM(0.05)" "RMSProp(0.001)" "RMSProp(0.005)" "RMSProp(0.05)" "BFGS()" "LBFGS()" "ADAM + BFGS"], legend=:bottomleft, linecolor=["#2660A4" "#4CD0F4" "#FEC32F" "#F763CD" "#44BD79" "#831894" "#A6ED18" "#980000" "#FF912B"])
savefig("pnp_loss_over_iterations.png")

@show loss_1[end], loss_2[end], loss_3[end], loss_4[end], loss_5[end], loss_6[end], loss_7[end], loss_8[end], loss_9[end]