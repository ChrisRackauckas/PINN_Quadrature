
#/////////////////////////////////////////////////////////////////////////////////
# INTERFACE TO RUN MUPLTIPLE EXAMPLES WITH DIFFERENT STRATEGIES / SETTINGS
#/////////////////////////////////////////////////////////////////////////////////
using Plots
# Import all the examples
include("./nernst_planck_3D.jl")
include("./level_set.jl")
include("./allen_cahn.jl")
include("./hamilton_jacobi.jl")


# Settings:
maxIters   = 5  #number of iterations


strategies = [NeuralPDE.QuadratureTraining(algorithm = CubaCuhre(), reltol = 1e-8, abstol = 1e-8, maxiters = 100),
              NeuralPDE.QuadratureTraining(algorithm = HCubatureJL(), reltol = 1e-8, abstol = 1e-8, maxiters = 100),
              NeuralPDE.QuadratureTraining(algorithm = CubatureJLh(), reltol = 1e-8, abstol = 1e-8, maxiters = 100),
              NeuralPDE.QuadratureTraining(algorithm = CubatureJLp(), reltol = 1e-8, abstol = 1e-8, maxiters = 100),
              NeuralPDE.GridTraining(),
              NeuralPDE.StochasticTraining(),
              NeuralPDE.QuasiRandomTraining(sampling_method = UniformSample(),
                                                     number_of_points = 100,
                                                     number_of_minibatch = 100)]

strategies_short_name = ["CubaCuhre", "HCubatureJL", "CubatureJLh", "CubatureJLp", "GridTraining", "StochasticTraining",
                         "QuasiRandomTraining"]

minimizers = [GalacticOptim.ADAM(0.01), GalacticOptim.BFGS(), GalacticOptim.LBFGS()]

minimizers_short_name = ["ADAM", "BFGS", "LBFGS"]

# Run models
numeric_res = Dict()
prediction_res =  Dict()
benchmark_res = Dict()
losses_res =  Dict()
domains = Dict()

for strat=1:length(strategies) # strategy
      for min =1:length(minimizers) # minimizer
            t_0 = time_ns()
            #println(string(strategies[strat], minimizers[min]))
            res = level_set(strategies[strat], minimizers[min], maxIters)
            push!(losses_res, string(strat,min)     => res[1])
            push!(prediction_res, string(strat,min) => res[2])
            push!(numeric_res, string(strat,min)    => res[3])
            push!(domains, string(strat,min)        => res[4])
            push!(benchmark_res, string(strat,min)  => res[5])
            print(string("Training time ", strategies[strat], " ", minimizers[min], " = ",(res[5])))
      end
end



#////////////////////////////////////////
# ANALYSIS OF RESULTS: LEVEL-SET
#////////////////////////////////////////

## Time Benchmark
benchmark_res_name = Dict()
for strat=1:length(strategies) # strategy
      for min =1:length(minimizers)
            push!(benchmark_res_name, string(strategies_short_name[strat], " + " , minimizers_short_name[min]) => benchmark_res[string(strat,min)])
      end
end
Plots.bar(collect(keys(benchmark_res_name)), collect(values(benchmark_res_name)), title = string("Level_set"), xrotation = 90)
Plots.savefig("Level_set.pdf")


## Convergence
#Plotting the first strategy with the first minimizer out from the loop to initialize the canvas
current_label = string("Strategy: ", strategies[1], "  Minimizer: ", minimizers[1])
Plots.plot(1:(maxIters + 1), losses_res["11"], yaxis=:log10, title = string("Level_set"), ylabel = "log(loss)", legend = true)
for strat=2:5#length(strategies) # strategy
      for min =1:length(minimizers) # minimizer
            # Learning curve plots with different strategies, minimizer
            current_label = string("Strategy: ", strategies[strat], "  Minimizer: ", minimizers[min])
            Plots.plot!(1:(maxIters + 1), losses_res[string(strat,min)], yaxis=:log10, title = string("Level_set"), ylabel = "log(loss)", legend = true)
      end
end
Plots.savefig("Level_set_loss.pdf")


## Comparison Predicted solution vs Numerical solution
integrals = Dict()

to_print = [] #["11", "32"]

for strat=1:5 #length(strategies) # strategy
      for min =1:length(minimizers)
            u_predict = prediction_res[string(strat,min)][1]
            u_real = numeric_res[string(strat,min)][1]

            # dimensions
            ts = collect(domains[string(strat,min)][1])
            xs = collect(domains[string(strat,min)][2])
            ys = collect(domains[string(strat,min)][3])
            zs = collect(domains[string(strat,min)][4])
            XS = [xs,ys,zs]

            diff_u = abs.(u_predict .- u_real)

            if string(strat,min) ∈ to_print
                  """
                  p1 = Plots.plot(XS, ts, u_real, linetype=:contourf,title = "analytic");
                  p2 = Plots.plot(u_predict[1,:,:,1], linetype=:contourf, title = "predict");
                  p3 = Plots.plot(XS, ts, diff_u,linetype=:contourf,title = "error");
                  savefig(p1,"p1.pdf")
                  savefig(p2,"p2.pdf")
                  savefig(p3,"p3.pdf")
                  """
            end

            integral = sum(diff_u)[1]
            push!(integrals, string(strategies_short_name[strat], " + " , minimizers_short_name[min]) => integral)

      end
end

Plots.bar(collect(keys(integrals)), collect(values(integrals)), title = string("Nernst-Planck Error Integrals"), xrotation = 90)
savefig("Nernst-Planck_strategies_MAEs.pdf")
