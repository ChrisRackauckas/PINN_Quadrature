
#/////////////////////////////////////////////////////////////////////////////////
# INTERFACE TO RUN MUPLTIPLE EXAMPLES WITH DIFFERENT STRATEGIES / SETTINGS
#/////////////////////////////////////////////////////////////////////////////////

# Import all the examples
include("./nernst_planck_3D.jl")
include("./level_set.jl")

# Settings:
maxIters   = 300  #number of iterations


strategies = [NeuralPDE.QuadratureTraining(algorithm = CubaCuhre(), reltol = 1e-8, abstol = 1e-8, maxiters = 100),
              NeuralPDE.QuadratureTraining(algorithm = HCubatureJL(), reltol = 1e-8, abstol = 1e-8, maxiters = 100),
              NeuralPDE.QuadratureTraining(algorithm = CubatureJLh(), reltol = 1e-8, abstol = 1e-8, maxiters = 100),
              NeuralPDE.QuadratureTraining(algorithm = CubatureJLp(), reltol = 1e-8, abstol = 1e-8, maxiters = 100),
              NeuralPDE.GridTraining(),
              NeuralPDE.StochasticTraining(),
              NeuralPDE.QuasiRandomTraining(sampling_method = UniformSample(),
                                                     number_of_points = 100,
                                                     number_of_minibatch = 100)]

minimizers = [GalacticOptim.ADAM(0.01), GalacticOptim.BFGS(), GalacticOptim.LBFGS()]

# Run models
numeric_res, prediction_res, benchmark_res, losses_res, domains, time = zeros((3,3))

for strat=1:length(strategies) # strategy
      for min =1:length(minimizers) # minimizer
            t_0 = time_ns()
            println(string(strategies[strat], minimizers[min]))
            res = nernst_planck(strategies[strat], minimizers[min], maxIters)
            losses_res[strat,min]     = res[1]
            prediction_res[strat,min] = res[2]
            numeric_res[strat,min]    = res[3]
            domains[strat,min]        = res[4]
            benchmark_res[strat,min]  = res[5]
            print(string("Training time ", strategies[strat], " ", minimizers[min], " = ",(benchmark_res[end])))
      end
end



#////////////////////////////
# ANALYSIS OF RESULTS
#////////////////////////////


current_label = string("Strategy: ", strategies[1], "  Minimizer: ", minimizers[1])
Plots.plot(1:(maxIters + 1), benchmark_res[1,1], yaxis=:log, title = string("Nernst-Planck"), ylabel = "log(loss)", legend = true)

for strat=2:length(strategies) # strategy
      for min =2:length(minimizers) # minimizer
            # Time benchmarks

            current_label = string("Strategy: ", strategies[strat], "  Minimizer: ", minimizers[min])
            Plots.plot!(1:(maxIters + 1), benchmark_res[strat,min], yaxis=:log, label = current_label, title = string("Nernst-Planck"), ylabel = "log(loss)", legend = true)
            # Comparison between multiple strategies: maxIters fixed
            #   Here you can access the u_predic, u_numeric arrays of all the examples trained with different the strategies / minimzers.
            #   numeric_res[i,j] ...  prediction_res[i,j]
            # Make plots and comparisons
      end
end

savefig("Nernst-Planck_time.pdf")

current_label = string("Strategy: ", strategies[1], "  Minimizer: ", minimizers[1])
Plots.plot(1:(maxIters + 1), losses_res[1,1], yaxis=:log, title = string("Nernst-Planck"), ylabel = "log(loss)", legend = true)

for strat=2:length(strategies) # strategy
      for min =2:length(minimizers) # minimizer
            # Learning curve plots with different strategies, minimizer
            current_label = string("Strategy: ", strategies[strat], "  Minimizer: ", minimizers[min])
            lossPlot = Plots.plot(1:(maxIters + 1), losses_res[strat,min], yaxis=:log, title = string("Nernst-Planck"), ylabel = "log(loss)", legend = true)


      end
end

savefig("Nernst-Planck_loss.pdf")
