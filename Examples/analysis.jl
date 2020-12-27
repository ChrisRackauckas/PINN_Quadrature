
#/////////////////////////////////////////////////////////////////////////////////
# INTERFACE TO RUN MUPLTIPLE EXAMPLES WITH DIFFERENT STRATEGIES / SETTINGS
#/////////////////////////////////////////////////////////////////////////////////

# Import all the examples
include("./nernst_planck_3D.jl")
include("./level_set.jl")

# Settings:
maxIters   = 3  #number of iterations


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
            res = nernst_planck(strategies[strat], minimizers[min], maxIters)
            push!(losses_res, string(strat,min)     => res[1])
            push!(prediction_res, string(strat,min) => res[2])
            push!(numeric_res, string(strat,min)    => res[3])
            push!(domains, string(strat,min)        => res[4])
            push!(benchmark_res, string(strat,min)  => res[5])
            print(string("Training time ", strategies[strat], " ", minimizers[min], " = ",(res[5])))
      end
end



#////////////////////////////////////////
# ANALYSIS OF RESULTS: NERNST-PLANCK 3D
#////////////////////////////////////////

# Time Benchmark
#----------------------------
benchmark_res_name = Dict()
for strat=1:5#length(strategies) # strategy
      for min =1:length(minimizers)
            push!(benchmark_res_name, string(strategies_short_name[strat], " + " , minimizers_short_name[min]) => benchmark_res[string(strat,min)])
      end
end
Plots.bar(collect(keys(benchmark_res_name)), collect(values(benchmark_res_name)), title = string("Nernst-Planck"), xrotation = 90, label = "")
savefig("Nernst-Planck_time.pdf")


# Convergence
#----------------------------
#Plotting the first strategy with the first minimizer out from the loop to initialize the canvas
current_label = string("Strategy: ", strategies[1], "  Minimizer: ", minimizers[1])
Plots.plot(1:(maxIters + 1), losses_res["11"], yaxis=:log10, title = string("Nernst-Planck"), ylabel = "log(loss)", legend = true)
for strat=2:5#length(strategies) # strategy
      for min =2:length(minimizers) # minimizer
            # Learning curve plots with different strategies, minimizer
            current_label = string("Strategy: ", strategies[strat], "  Minimizer: ", minimizers[min])
            Plots.plot!(1:(maxIters + 1), losses_res[string(strat,min)], yaxis=:log10, title = string("Nernst-Planck"), ylabel = "log(loss)", legend = true)
      end
end
savefig("Nernst-Planck_loss.pdf")


# U-Predict <=> U-Numeric comparison
#---------------------------

p1 = plot(xs, ts, u_real, linetype=:contourf,title = "analytic");
p2 = plot(xs, ts, u_predict, linetype=:contourf,title = "predict $name");
p3 = plot(xs, ts, diff_u,linetype=:contourf,title = "error");
