using NeuralPDE
using Quadrature, Cubature, Cuba
using Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Plots
using PyPlot
using DelimitedFiles
using QuasiMonteCarlo
using JLD

function allen_cahn(strategy, minimizer, maxIters)

    ##  DECLARATIONS
    @parameters  t x1 x2 x3 x4
    @variables   u(..)

    Dt = Differential(t)
    Dxx1 = Differential(x1)^2
    Dxx2 = Differential(x2)^2
    Dxx3 = Differential(x3)^2
    Dxx4 = Differential(x4)^2


    # Discretization
    tmax         = 1.0
    x1width      = 1.0
    x2width      = 1.0
    x3width      = 1.0
    x4width      = 1.0

    tMeshNum     = 10
    x1MeshNum    = 10
    x2MeshNum    = 10
    x3MeshNum    = 10
    x4MeshNum    = 10

    dt   = tmax/tMeshNum
    dx1  = x1width/x1MeshNum
    dx2  = x2width/x2MeshNum
    dx3  = x3width/x3MeshNum
    dx4  = x4width/x4MeshNum

    domains = [t ∈ IntervalDomain(0.0,tmax),
               x1 ∈ IntervalDomain(0.0,x1width),
               x2 ∈ IntervalDomain(0.0,x2width),
               x3 ∈ IntervalDomain(0.0,x3width),
               x4 ∈ IntervalDomain(0.0,x4width)]

    ts  = 0.0 : dt : tmax
    x1s = 0.0 : dx1 : x1width
    x2s = 0.0 : dx2 : x2width
    x3s = 0.0 : dx3 : x3width
    x4s = 0.0 : dx4 : x4width

    # Operators
    Δu = Dxx1(u(t,x1,x2,x3,x4)) + Dxx2(u(t,x1,x2,x3,x4)) + Dxx3(u(t,x1,x2,x3,x4)) + Dxx4(u(t,x1,x2,x3,x4)) # Laplacian


    # Equation
    eq = Dt(u(t,x1,x2,x3,x4)) - Δu - u(t,x1,x2,x3,x4) + u(t,x1,x2,x3,x4)*u(t,x1,x2,x3,x4)*u(t,x1,x2,x3,x4) ~ 0  #ALLEN CAHN EQUATION

    initialCondition =  1/(2 + 0.4 * (x1*x1 + x2*x2 + x3*x3 + x4*x4)) # see PNAS paper

    bcs = [u(0,x1,x2,x3,x4) ~ initialCondition]  #from literature

    ## NEURAL NETWORK
    n = 20   #neuron number

    chain = FastChain(FastDense(5,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1))   #Neural network from Flux library

    indvars = [t,x1,x2,x3,x4]   #phisically independent variables
    depvars = [u]       #dependent (target) variable

    dim = length(domains)

    losses = []
    error = []
    times = []

    dx_err = 0.1

    error_strategy = GridTraining(dx_err)

    phi = NeuralPDE.get_phi(chain)
    derivative = NeuralPDE.get_numeric_derivative()
    initθ = DiffEqFlux.initial_params(chain)

    _pde_loss_function = NeuralPDE.build_loss_function(eq,indvars,depvars,
                                             phi,derivative,chain,initθ,error_strategy)

    bc_indvars = NeuralPDE.get_variables(bcs,indvars,depvars)
    _bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars,
                                              phi,derivative,chain,initθ,error_strategy,
                                              bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]

    train_sets = NeuralPDE.generate_training_sets(domains,dx_err,[eq],bcs,indvars,depvars)
    train_domain_set, train_bound_set = train_sets


    pde_loss_function = NeuralPDE.get_loss_function([_pde_loss_function],
                                          train_domain_set,
                                          error_strategy)

    bc_loss_function = NeuralPDE.get_loss_function(_bc_loss_functions,
                                         train_bound_set,
                                         error_strategy)

    function loss_function_(θ,p)
        return pde_loss_function(θ) + bc_loss_function(θ)
    end

    cb_ = function (p,l)
        deltaT_s = time_ns() #Start a clock when the callback begins, this will evaluate questo misurerà anche il calcolo degli uniform error

        ctime = time_ns() - startTime - timeCounter #This variable is the time to use for the time benchmark plot
        append!(times, ctime/10^9) #Conversion nanosec to seconds
        append!(losses, l)
        append!(error, pde_loss_function(p) + bc_loss_function(p))
        println(length(losses), " Current loss is: ", l, " uniform error is, ",  pde_loss_function(p) + bc_loss_function(p))

        timeCounter = timeCounter + time_ns() - deltaT_s #timeCounter sums all delays due to the callback functions of the previous iterations

        #if (ctime/10^9 > time) #if I exceed the limit time I stop the training
        #    return true #Stop the minimizer and continue from line 142
        #end

        return false
    end

    pde_system = PDESystem(eq, bcs, domains, indvars, depvars)

    discretization = NeuralPDE.PhysicsInformedNN(chain, strategy)
    prob = NeuralPDE.discretize(pde_system,discretization)

    timeCounter = 0.0
    startTime = time_ns() #Fix initial time (t=0) before starting the training
    res = GalacticOptim.solve(prob, minimizer, cb = cb_, maxiters=maxIters)

    phi = discretization.phi

    params = res.minimizer

    # Model prediction
    domain = [ts,x1s,x2s,x3s,x4s]

    u_predict = [reshape([first(phi([t,x1,x2,x3,x4],res.minimizer)) for x1 in x1s for x2 in x2s for x3 in x3s for x4 in x4s], (length(x1s),length(x2s), length(x3s),length(x4s))) for t in ts]  #matrix of model's prediction

    return [error, params, domain, times, losses]
end
