using NeuralPDE
using Quadrature, Cubature, Cuba
using Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Plots
using PyPlot
using DelimitedFiles
using QuasiMonteCarlo
using JLD

print("Precompiling Done")

function level_set(strategy, minimizer, time)

    ##  DECLARATIONS
    @parameters  t x y
    @variables   u(..)
    maxIters = 10000

    Dt = Differential(t)
    Dx = Differential(x)
    Dy = Differential(y)

    # Discretization
    xwidth      = 1.0      #ft
    ywidth      = 1.0
    tmax        = 1.0      #min
    xScale      = 1.0
    yScale      = 1.0
    xMeshNum    = 10
    yMeshNum    = 10
    tMeshNum    = 10
    dx  = xwidth/xMeshNum
    dy  = ywidth/yMeshNum
    dt  = tmax/tMeshNum

    domains = [t ∈ IntervalDomain(0.0,tmax),
               x ∈ IntervalDomain(0.0,xwidth),
               y ∈ IntervalDomain(0.0,ywidth)]

    xs = 0.0 : dx : xwidth
    ys = 0.0 : dy : ywidth
    ts = 0.0 : dt : tmax

    # Definitions
    x0    = 0.5
    y0    = 0.5
    Uwind = [0.0, 2.0]  #wind vector

    # Operators
    gn   = (Dx(u(t,x,y))^2 + Dy(u(t,x,y))^2)^0.5  #gradient's norm
    ∇u   = [Dx(u(t,x,y)), Dy(u(t,x,y))]
    n    = ∇u/gn              #normal versor
    #U    = ((Uwind[1]*n[1] + Uwind[2]*n[2])^2)^0.5 #inner product between wind and normal vector

    R0 = 0.112471
    ϕw = 0#0.156927*max((0.44*U)^0.04086,1.447799)
    ϕs = 0
    S  = R0*(1 + ϕw + ϕs)

    # Equation
    eq = Dt(u(t,x,y)) + S*gn ~ 0  #LEVEL SET EQUATION

    initialCondition = (xScale*(x - x0)^2 + (yScale*(y - y0)^2))^0.5 - 0.2   #Distance from ignition

    bcs = [u(0,x,y) ~ initialCondition]  #from literature


    ## NEURAL NETWORK
    n = 16   #neuron number

    chain = FastChain(FastDense(3,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1))   #Neural network from Flux library

    indvars = [t,x,y]   #phisically independent variables
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
        if (ctime > time) #if I exceed the limit time I stop the training
            return true #Stop the minimizer and continue from line 142
        end

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
    domain = [ts, xs, ys]

    u_predict = [reshape([first(phi([t,x,y],res.minimizer)) for x in xs for y in ys], (length(xs),length(ys))) for t in ts]  #matrix of model's prediction

    return [error, params, domain, times] #add numeric solution
end

#level_set(NeuralPDE.QuadratureTraining(algorithm = CubaCuhre(), reltol = 1e-8, abstol = 1e-8, maxiters = 100), GalacticOptim.ADAM(0.01), 500)
