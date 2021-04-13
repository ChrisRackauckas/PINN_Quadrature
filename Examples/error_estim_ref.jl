using NeuralPDE
using Quadrature, Cubature, Cuba
using Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Plots
#using PyPlot
import Base.Broadcast
import ModelingToolkit: value, nameof, toexpr, build_expr, expand_derivatives


strategy = NeuralPDE.QuadratureTraining(algorithm = CubaCuhre(), reltol = 1e-8, abstol = 1e-8, maxiters = 100)
minimizer = GalacticOptim.ADAM(0.01)
maxIters = 30

#function hamilton_jacobi(strategy, minimizer, maxIters)

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
eq = Dt(u(t,x1,x2,x3,x4)) + Δu - λ*sum(∇u.^2) ~ 0  #HAMILTON-JACOBI-BELLMAN EQUATION

terminalCondition =  log((1 + x1*x1 + x2*x2 + x3*x3 + x4*x4)/2) # see PNAS paper

bcs = [u(tmax,x1,x2,x3,x4) ~ terminalCondition]  #PNAS paper again

## NEURAL NETWORK
n = 20   #neuron number

chain = FastChain(FastDense(5,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1))   #Neural network from Flux library

discretization = NeuralPDE.PhysicsInformedNN(chain, strategy = strategy) #Discretization used for training
ref_dscr = NeuralPDE.PhysicsInformedNN(chain, strategy = GridTraining()) #Discretization used for error estimation

indvars = [t,x1,x2,x3,x4]   #phisically independent variables
depvars = [u]       #dependent (target) variable

dim = length(domains)

losses = []
cb = function (p,l)     #loss function handling

    ref_metric=strategy

    phi = NeuralPDE.get_phi(chain)
    derivative = NeuralPDE.get_numeric_derivative()
    initθ = DiffEqFlux.initial_params(chain)

    θ = gensym("θ")

    function _transform_expression(ex,dict_indvars,dict_depvars, initθ, strategy)
        _args = ex.args
        for (i,e) in enumerate(_args)
            if !(e isa Expr)
                if e in keys(dict_depvars)
                    depvar = _args[1]
                    num_depvar = dict_depvars[depvar]
                    indvars = _args[2:end]
                    cord = :cord
                    ex.args = if length(dict_depvars) == 1
                        [:u, cord, :($θ), :phi]
                    else
                        [:u, cord, Symbol(:($θ),num_depvar), Symbol(:phi,num_depvar)]
                    end
                    break
                elseif e isa ModelingToolkit.Differential
                    derivative_variables = Symbol[]
                    order = 0
                    while (_args[1] isa ModelingToolkit.Differential)
                        order += 1
                        push!(derivative_variables, toexpr(_args[1].x))
                        _args = _args[2].args
                    end
                    depvar = _args[1]
                    num_depvar = dict_depvars[depvar]
                    indvars = _args[2:end]
                    cord = :cord
                    dim_l = length(indvars)
                    εs = [get_ε(dim_l,d) for d in 1:dim_l]
                    undv = [dict_indvars[d_p] for d_p  in derivative_variables]
                    εs_dnv = [εs[d] for d in undv]
                    ex.args = if length(dict_depvars) == 1
                        [:derivative, :phi, :u, cord, εs_dnv, order, :($θ)]
                    else
                        [:derivative, Symbol(:phi,num_depvar), :u, cord, εs_dnv, order, Symbol(:($θ),num_depvar)]
                    end
                    break
                end
            else
                ex.args[i] = _transform_expression(ex.args[i],dict_indvars,dict_depvars,initθ,strategy)
            end
        end
        return ex
    end

    function transform_expression(ex,dict_indvars,dict_depvars, initθ, strategy)
        if ex isa Expr
            ex = _transform_expression(ex,dict_indvars,dict_depvars,initθ,strategy)
        end
        return ex
    end

    get_dict_vars(vars) = Dict( [Symbol(v) .=> i for (i,v) in enumerate(vars)])
    function get_vars(indvars_, depvars_)
        depvars = [nameof(value(d)) for d in depvars_]
        indvars = [nameof(value(i)) for i in indvars_]
        dict_indvars = get_dict_vars(indvars)
        dict_depvars = get_dict_vars(depvars)
        return depvars,indvars,dict_indvars,dict_depvars
    end

    indvars = [t,x1,x2,x3,x4]   #phisically independent variables
    depvars = [u]
    depvars,indvars,dict_indvars,dict_depvars = get_vars(indvars, depvars)

    eq_lhs = isequal(expand_derivatives(eq.lhs), 0) ? eq.lhs : expand_derivatives(eq.lhs)
    eq_rhs = isequal(expand_derivatives(eq.rhs), 0) ? eq.rhs : expand_derivatives(eq.rhs)

    left_expr = transform_expression(toexpr(eq_lhs),dict_indvars,dict_depvars,initθ,ref_metric)
    right_expr = transform_expression(toexpr(eq_rhs),dict_indvars,dict_depvars,initθ,ref_metric)
    left_expr = Broadcast.__dot__(left_expr)
    right_expr = Broadcast.__dot__(right_expr)
    loss_func = :($left_expr .- $right_expr)

    function loss_function_(θ,p)
        return pde_loss_function(θ) + bc_loss_function(θ)
    end

    #append!(losses, loss_func)
    println("Current loss is: $l  uniform error is: $loss_func")
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

#return [losses, u_predict, u_predict, domain, training_time] #add numeric solution


losses, u_predict, u_predict, domain, training_time = hamilton_jacobi(NeuralPDE.QuadratureTraining(algorithm = CubaCuhre(), reltol = 1e-8, abstol = 1e-8, maxiters = 100),
                                                      GalacticOptim.ADAM(0.01), 500)
