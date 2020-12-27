using NeuralPDE
using Quadrature, Cubature, Cuba
using Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Plots
using PyPlot
using DelimitedFiles
using QuasiMonteCarlo

print("Precompiling Done")

res = nernst_planck(NeuralPDE.QuadratureTraining(algorithm = CubaCuhre(),reltol = 1e-8, abstol = 1e-8, maxiters = 100), GalacticOptim.ADAM(0.01), 3)

function nernst_planck(strategy, minimizer, maxIters)

    ##  DECLARATIONS
    @parameters t x y z
    @variables c(..)
    @derivatives Dt'~t
    @derivatives Dx'~x
    @derivatives Dy'~y
    @derivatives Dz'~z
    @derivatives Dxx''~x
    @derivatives Dyy''~y
    @derivatives Dzz''~z

    ## DOMAINS AND OPERATORS

    # Discretization
    xwidth      = 1.0
    ywidth      = 1.0
    zwidth      = 1.0
    tmax        = 1.0
    xMeshNum    = 10
    yMeshNum    = 10
    zMeshNum    = 10
    tMeshNum    = 10
    dx  = xwidth/xMeshNum
    dy  = ywidth/yMeshNum
    dz  = zwidth/zMeshNum
    dt  = tmax/tMeshNum

    domains = [t ∈ IntervalDomain(0.0,tmax),
               x ∈ IntervalDomain(0.0,xwidth),
               y ∈ IntervalDomain(0.0,ywidth),
               z ∈ IntervalDomain(0.0,zwidth)]

    xs = 0.0 : dx : xwidth
    ys = 0.0 : dy : ywidth
    zs = 0.0 : dz : zwidth
    ts = 0.0 : dt : tmax

    # Constants
    D = 1  #dummy
    ux = 10 #dummy
    uy = 10 #dummy
    uz = 10 #dummy

    # Operators
    #grad_c = [Dx(c(t,x,y,z)), Dy(c(t,x,y,z)), Dz(c(t,x,y,z))]
    div = - D*(Dxx(c(t,x,y,z)) + Dyy(c(t,x,y,z)) + Dzz(c(t,x,y,z)))
          + (ux*Dx(c(t,x,y,z)) + uy*Dy(c(t,x,y,z)) + uz*Dz(c(t,x,y,z)))

    # Equation
    eq = Dt(c(t,x,y,z)) + div ~ 0      #NERNST-PLANCK EQUATION

    # Boundary conditions
    bcs = [c(0,x,y,z) ~ 0]

    ## NEURAL NETWORK
    n = 16   #neuron number

    chain = FastChain(FastDense(4,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1))   #Neural network from Flux library

    discretization = NeuralPDE.PhysicsInformedNN(chain,strategy = strategy)

    indvars = [t,x,y,z]   #independent variables
    depvars = [c]       #dependent (target) variable

    dim = length(domains)

    losses = []
    cb = function (p,l)     #loss function handling
        println("Current loss is: $l")
        append!(losses, l)
        return false
    end

    pde_system = PDESystem(eq, bcs, domains, indvars, depvars)
    prob = discretize(pde_system, discretization)

    t_0 = time_ns()

    res = GalacticOptim.solve(prob, minimizer; cb = cb, maxiters=maxIters) #allow_f_increase = false,

    t_f = time_ns()
    training_time = (t_f-t_0)/10^9
    #initθ = res.minimizer

    #pars = open(readdlm,"/np_params.txt") #to import parameters from previous training
    #pars

    phi = discretization.phi

    domain = [ts, xs, ys, zs]

    u_predict  = [reshape([phi([t,x,y,z],res.minimizer) for t in ts for x in xs for y in ys for z in zs],
                 (length(ts),length(xs),length(ys),length(zs)))]


    return [losses, u_predict, u_predict, domain, training_time] #add numeric solution
end
