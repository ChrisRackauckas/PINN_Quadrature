using NeuralPDE
using Quadrature, Cubature, Cuba
using Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Plots
using PyPlot
using DelimitedFiles

print("Precompiling Done")

level_set(NeuralPDE.QuadratureTraining(algorithm = CubaCuhre(), reltol = 1e-8, abstol = 1e-8, maxiters = 100),GalacticOptim.ADAM(0.01), 3000)

function level_set(strategy, minimizer, maxIters)

    ##  DECLARATIONS
    @parameters t x y θ
    @variables u(..)
    @derivatives Dt'~t
    @derivatives Dx'~x
    @derivatives Dy'~y

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

    # Definitions
    gn   = (Dx(u(t,x,y,θ))^2 + Dy(u(t,x,y,θ))^2)^0.5 #gradient's norm
    S = 1
    eq = Dt(u(t,x,y,θ)) + S*gn ~ 0   #LEVEL SET EQUATION

    initialCondition = (xScale*x^2 + (yScale*y^2))^0.5 - 0.2   #Distance from ignition

    bcs = [u(0,x,y,θ) ~ initialCondition]  #from literature


    ## NEURAL NETWORK
    n = 16   #neuron number

    chain = FastChain(FastDense(3,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1))   #Neural network from Flux library

    q_strategy = NeuralPDE.QuadratureTraining(algorithm =CubaCuhre(),reltol=1e-8,abstol=1e-8,maxiters=100)  #Training strategy

    discretization = NeuralPDE.PhysicsInformedNN([dt,dx,dy],chain,strategy = q_strategy)


    indvars = [t,x,y]   #phisically independent variables
    depvars = [u]       #dependent (target) variable

    dim = length(domains)

    losses = []
    cb = function (p,l)     #loss function handling
        println("Current loss is: $l")
        append!(losses, l)
        return false
    end

    pde_system = PDESystem(eq, bcs, domains, indvars, depvars)
    prob = discretize(pde_system, discretization)

    a_1 = time_ns()

    res = GalacticOptim.solve(prob, minimizer = minimizer, cb = cb, maxiters=maxIters) #allow_f_increase = false,

    initθ = res.minimizer

    discretization2 = NeuralPDE.PhysicsInformedNN([dt,dx,dy],chain, initθ; strategy = strategy)   #Second learning phase, lower learning parameter
    initθ == discretization2.initθ
    prob2 = NeuralPDE.discretize(pde_system,discretization2)
    res2 = GalacticOptim.solve(prob2, GalacticOptim.ADAM(0.001), cb = cb, maxiters=4000)
    b_1 = time_ns()
    print(string("Training time = ",(b_1-a_1)/10^9))
    initθ2 = res2.minimizer


    par = open(readdlm,"/Julia_implementation/LevelSetEq/params_level_set_stage_final_4800iter.txt") #to import parameters from previous training (change also line 227 accordingly)
    par


    phi = discretization.phi

    printBCSComp = true     #prints initial condition comparison and training loss plot

    xs = 0.0 : dx : xwidth
    ys = 0.0 : dy : ywidth

    ts = 1 : dt : tmax

    u_predict = [reshape([first(phi([t,x,y],res2.minimizer)) for x in xs for y in ys], (length(xs),length(ys))) for t in ts]  #matrix of model's prediction

    maxlim = maximum(maximum(u_predict[t]) for t = 1:length(ts))
    minlim = minimum(minimum(u_predict[t]) for t = 1:length(ts))

        trainingPlot = Plots.plot(1:(maxIters + 1), losses, yaxis=:log, title = string("Training time = 270 s",
            "\\n Iterations: ", maxIters, "   NN: 3>16>1"), ylabel = "log(loss)", legend = false) #loss plot

end
