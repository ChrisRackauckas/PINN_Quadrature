using NeuralPDE
using Quadrature, Cubature, Cuba
using Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Plots
using PyPlot


#losses, u_predict, u_predict, domain, training_time = hamilton_jacobi(NeuralPDE.QuadratureTraining(algorithm = CubaCuhre(), reltol = 1e-8, abstol = 1e-8, maxiters = 1000), GalacticOptim.ADAM(0.01), 100)


function hamilton_jacobi(strategy, minimizer, maxIters)

    ##  DECLARATIONS
    @parameters  t x1 x2 x3 x4
    @variables   u(..)

    @derivatives Dt'~t

    @derivatives Dx1'~x1
    @derivatives Dx2'~x2
    @derivatives Dx3'~x3
    @derivatives Dx4'~x4

    @derivatives Dxx1''~x1
    @derivatives Dxx2''~x2
    @derivatives Dxx3''~x3
    @derivatives Dxx4''~x4


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

    λ = 1.0f0

    # Operators
    Δu = Dxx1(u(t,x1,x2,x3,x4)) + Dxx2(u(t,x1,x2,x3,x4)) + Dxx3(u(t,x1,x2,x3,x4)) + Dxx4(u(t,x1,x2,x3,x4)) # Laplacian
    ∇u = [Dx1(u(t,x1,x2,x3,x4)), Dx2(u(t,x1,x2,x3,x4)),Dx3(u(t,x1,x2,x3,x4)),Dx4(u(t,x1,x2,x3,x4))]

    # Equation
    eq = Dt(u(t,x1,x2,x3,x4)) + Δu - λ*sum(∇u.^2) ~ 0  #LEVEL SET EQUATION

    initialCondition =  log((1 + x1*x1 + x2*x2 + x3*x3 + x4*x4)/2) # see PNAS paper

    bcs = [u(tmax,x1,x2,x3,x4) ~ initialCondition]  #PNAS paper again

    ## NEURAL NETWORK
    n = 20   #neuron number

    chain = FastChain(FastDense(5,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1))   #Neural network from Flux library

    discretization = NeuralPDE.PhysicsInformedNN(chain, strategy = strategy)

    indvars = [t,x1,x2,x3,x4]   #phisically independent variables
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

    t_0 = time_ns()

    res = GalacticOptim.solve(prob, minimizer; cb = cb, maxiters=maxIters) #allow_f_increase = false,

    t_f = time_ns()
    training_time = (t_f - t_0)/10^9
    #print(string("Training time = ",(t_f - t_0)/10^9))

    phi = discretization.phi

    # Model prediction
    domain = [ts,x1s,x2s,x3s,x4s]

    u_predict = [reshape([first(phi([t,x1,x2,x3,x4],res.minimizer)) for x1 in x1s for x2 in x2s for x3 in x3s for x4 in x4s], (length(x1s),length(x2s), length(x3s),length(x4s))) for t in ts]  #matrix of model's prediction

    return [losses, u_predict, u_predict, domain, training_time] #add numeric solution
end
