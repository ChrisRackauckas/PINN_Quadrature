using NeuralPDE
using Quadrature, Cubature, Cuba
using Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Plots
#using PyPlot
import Base.Broadcast
import ModelingToolkit: value, nameof, toexpr, build_expr, expand_derivatives
using RuntimeGeneratedFunctions, BenchmarkTools
RuntimeGeneratedFunctions.init(@__MODULE__)
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

    grid_res = 0.1
    ref_metric = NeuralPDE.GridTraining(grid_res)

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


    function build_symbolic_loss_function(eqs,indvars,depvars,
                                          dict_indvars,dict_depvars,
                                          phi, derivative, initθ)#, strategy; eq_params = SciMLBase.NullParameters(), param_estim = param_estim,default_p=default_p,
                                          #bc_indvars = indvars)
        bc_indvars = indvars
        strategy = ref_metric
        eq_params = initθ #SciMLBase.NullParameters()
        loss_function = loss_func #parse_equation(eqs,dict_indvars,dict_depvars,initθ,strategy)
        vars = :(cord, $θ, phi, derivative,u,p)
        ex = Expr(:block)
        if length(depvars) != 1
            θ_nums = Symbol[]
            phi_nums = Symbol[]
            for v in depvars
                num = dict_depvars[v]
                push!(θ_nums,:($(Symbol(:($θ),num))))
                push!(phi_nums,:($(Symbol(:phi,num))))
            end

            expr_θ = Expr[]
            expr_phi = Expr[]

            acum =  [0;accumulate(+, length.(initθ))]
            sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]

            for i in eachindex(depvars)
                push!(expr_θ, :($θ[$(sep[i])]))
                push!(expr_phi, :(phi[$i]))
            end

            vars_θ = Expr(:(=), build_expr(:tuple, θ_nums), build_expr(:tuple, expr_θ))
            push!(ex.args,  vars_θ)

            vars_phi = Expr(:(=), build_expr(:tuple, phi_nums), build_expr(:tuple, expr_phi))
            push!(ex.args,  vars_phi)
        end
        #Add an expression for parameter symbols
        #if param_estim == true && eq_params != SciMLBase.NullParameters()
        param_len = length(eq_params)
        last_indx =  [0;accumulate(+, length.(initθ))][end]
        params_symbols = Symbol[]
        expr_params = Expr[]
        for (i , eq_param) in enumerate(eq_params)
            push!(expr_params, :($θ[$(i+last_indx:i+last_indx)]))
            push!(params_symbols, Symbol(:($eq_param)))
        end
        params_eq = Expr(:(=), build_expr(:tuple, params_symbols), build_expr(:tuple, expr_params))
        push!(ex.args,  params_eq)
        #end

        #=if eq_params != SciMLBase.NullParameters() && param_estim == false
            params_symbols = Symbol[]
            expr_params = Expr[]
            for (i , eq_param) in enumerate(eq_params)
                push!(expr_params, :(ArrayInterface.allowed_getindex(p,$i:$i)))
                push!(params_symbols, Symbol(:($eq_param)))
            end
            params_eq = Expr(:(=), build_expr(:tuple, params_symbols), build_expr(:tuple, expr_params))
            push!(ex.args,  params_eq)
        end
=#
        #=if strategy isa QuadratureTraining

            indvars_ex = get_indvars_ex(bc_indvars)

            left_arg_pairs, right_arg_pairs = indvars,indvars_ex
            vcat_expr =  :(cord = vcat($(indvars...)))
            vcat_expr_loss_functions = Expr(:block,vcat_expr,loss_function) #TODO rename
            vars_eq = Expr(:(=), build_expr(:tuple, left_arg_pairs), build_expr(:tuple, right_arg_pairs))

        else
        =#
        indvars_ex = [:($:cord[[$i],:]) for (i, u) ∈ enumerate(indvars)]
        left_arg_pairs, right_arg_pairs = indvars,indvars_ex
        vars_eq = Expr(:(=), build_expr(:tuple, left_arg_pairs), build_expr(:tuple, right_arg_pairs))
        vcat_expr_loss_functions = loss_function #TODO rename
        #end

        let_ex = Expr(:let, vars_eq, vcat_expr_loss_functions)
        push!(ex.args,  let_ex)

        expr_loss_function = :(($vars) -> begin $ex end)
    end

    function build_loss_function(eqs,_indvars,_depvars, phi, derivative,initθ)#,strategy;bc_indvars=nothing,eq_params=SciMLBase.NullParameters(),param_estim=false,default_p=nothing)
        # dictionaries: variable -> unique number
        depvars,indvars,dict_indvars,dict_depvars = get_vars(_indvars, _depvars)
        bc_indvars = bc_indvars==nothing ? indvars : bc_indvars
        return build_loss_function(eqs,indvars,depvars,
                                   dict_indvars,dict_depvars,
                                   phi, derivative,initθ)#,strategy,
                                   #bc_indvars = bc_indvars, eq_params=eq_params,param_estim=param_estim,default_p=default_p)
    end

    function build_loss_function(eqs,indvars,depvars,
                                 dict_indvars,dict_depvars,
                                 phi, derivative, initθ)#, strategy;
                                 #bc_indvars = indvars,eq_params=SciMLBase.NullParameters(),param_estim=false,default_p=nothing)
         expr_loss_function = build_symbolic_loss_function(eqs,indvars,depvars,
                                                           dict_indvars,dict_depvars,
                                                           phi, derivative, initθ)#,strategy;
                                                           #bc_indvars = bc_indvars,eq_params = eq_params,param_estim=param_estim,default_p=default_p)
        function get_u()
            u = (cord, θ, phi)-> phi(cord, θ)
        end

        u = get_u()
        _loss_function = @RuntimeGeneratedFunction(expr_loss_function)
        loss_function = (cord, θ) -> begin
            _loss_function(cord, θ, phi, derivative, u, default_p)
        end
        return loss_function
    end

    _bc_loss_functions = [build_loss_function(bcs,indvars,depvars,
                                                  dict_indvars,dict_depvars,
                                                  phi, derivative, initθ)]#, strategy,eq_params=eq_params,param_estim=param_estim,default_p=default_p;
                                                  #bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]

    _pde_loss_functions = [build_loss_function(eq,indvars,depvars,
                                             dict_indvars,dict_depvars,
                                             phi, derivative, initθ)]#,strategy,eq_params=eq_params,param_estim=param_estim,default_p=default_p) for eq in eqs]


    function get_variables(eqs,_indvars::Array,_depvars::Array)
        depvars,indvars,dict_indvars,dict_depvars = get_vars(_indvars, _depvars)
        return get_variables(eqs,dict_indvars,dict_depvars)
    end

    function get_variables(eqs,dict_indvars,dict_depvars)
        bc_args = get_argument(eqs,dict_indvars,dict_depvars)
        return map(barg -> filter(x -> x isa Symbol, barg), bc_args)
    end

    function get_number(eqs,dict_indvars,dict_depvars)
        bc_args = get_argument(eqs,dict_indvars,dict_depvars)
        return map(barg -> filter(x -> x isa Number, barg), bc_args)
    end

    function find_thing_in_expr(ex::Expr, thing; ans = Expr[])
        for e in ex.args
            if e isa Expr
                if thing in e.args
                    push!(ans,e)
                end
                find_thing_in_expr(e,thing; ans=ans)
            end
        end
        return collect(Set(ans))
    end

    # Get arguments from boundary condition functions
    function get_argument(eqs,_indvars::Array,_depvars::Array)
        depvars,indvars,dict_indvars,dict_depvars = get_vars(_indvars, _depvars)
        get_argument(eqs,dict_indvars,dict_depvars)
    end
    function get_argument(eqs,dict_indvars,dict_depvars)
        exprs = toexpr.(eqs)
        vars = map(exprs) do expr
            _vars =  map(depvar -> find_thing_in_expr(expr,  depvar), collect(keys(dict_depvars)))
            f_vars = filter(x -> !isempty(x), _vars)
            map(x -> first(x), f_vars)
        end
        args_ = map(vars) do _vars
            map(var -> var.args[2:end] , _vars)
        end
        return first.(args_) #TODO for all arguments
    end


    function generate_training_sets(domains,dx,eqs,bcs,dict_indvars::Dict,dict_depvars::Dict)
        if dx isa Array
            dxs = dx
        else
            dxs = fill(dx,length(domains))
        end

        spans = [d.domain.lower:dx:d.domain.upper for (d,dx) in zip(domains,dxs)]
        dict_var_span = Dict([Symbol(d.variables) => d.domain.lower:dx:d.domain.upper for (d,dx) in zip(domains,dxs)])

        bound_args = get_argument(bcs,dict_indvars,dict_depvars)
        bound_vars = get_variables(bcs,dict_indvars,dict_depvars)

        dif = [Float32[] for i=1:size(domains)[1]]
        for _args in bound_args
            for (i,x) in enumerate(_args)
                if x isa Number
                    push!(dif[i],x)
                end
            end
        end
        cord_train_set = collect.(spans)
        bc_data = map(zip(dif,cord_train_set)) do (d,c)
            setdiff(c, d)
        end

        dict_var_span_ = Dict([Symbol(d.variables) => bc for (d,bc) in zip(domains,bc_data)])

        bcs_train_sets = map(bound_args) do bt
            span = map(b -> get(dict_var_span, b, b), bt)
            _set = Float32.(hcat(vec(map(points -> collect(points), Iterators.product(span...)))...))
        end

        pde_vars = get_variables(eqs,dict_indvars,dict_depvars)
        pde_args = get_argument(eqs,dict_indvars,dict_depvars)

        pde_train_set = Float32.(hcat(vec(map(points -> collect(points), Iterators.product(bc_data...)))...))

        pde_train_sets = map(pde_args) do bt
            span = map(b -> get(dict_var_span_, b, b), bt)
            _set = Float32.(hcat(vec(map(points -> collect(points), Iterators.product(span...)))...))
        end
        [pde_train_sets,bcs_train_sets]
    end


    train_sets = generate_training_sets(domains,grid_res,eq,bcs,get_dict_vars(indvars),get_dict_vars(depvars))
    train_domain_set, train_bound_set = train_sets


    pde_loss_function = NeuralPDE.get_loss_function(_pde_loss_function,
                        train_domain_set)

    bc_loss_function = NeuralPDE.get_loss_function(_bc_loss_functions,
                        train_bound_set)

    metric = pde_loss_function + bc_loss_function
    #append!(losses, loss_func)
    println("Current loss is: $l  uniform error is: $metric")
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
