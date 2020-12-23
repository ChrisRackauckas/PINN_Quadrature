
#/////////////////////////////////////////////////////////////////////////////////
# INTERFACE TO RUN MUPLTIPLE EXAMPLES WITH DIFFERENT STRATEGIES / SETTINGS
#/////////////////////////////////////////////////////////////////////////////////

# Import all the examples
include("./nernst_planck_3D.jl")
include("./level_set.jl")

# Settings:
maxIters   = 3000  #number of iterations

strategies = [NeuralPDE.QuadratureTraining(algorithm = CubaCuhre(), reltol = 1e-8, abstol = 1e-8, maxiters = 100),
              NeuralPDE.GridTraining(),
              NeuralPDE.StochasticTraining()]

minimizers = [GalacticOptim.ADAM(0.01), GalacticOptim.BFGS(), GalacticOptim.LBFGS()]

# Run models
numeric_res, prediction_res, benchmark_res, losses_res, domains, time = zeros((3,3))
for i=1:3
      for j=1:3
            t_0 = time_ns()
            res = nernst_planck_run(strategies[i], minimizers[j], maxIters)
            losses_res[i,j]     = res[1]
            prediction_res[i,j] = res[2]
            numeric_res[i,j]    = res[3]
            domains[i,j]        = res[4]
            benchmark_res[i,j]  = res[5]
            print(string("Training time ", strategies[i], " ", minimizers[j], " = ",(benchmark_res[end])))
      end
end



#////////////////////////////
# ANALYSIS OF RESULTS
#////////////////////////////


#ONLY A DRAFT!!
for i=1:3
      for j=1:3
            # Time benchmarks
            timePlot = Plots.plot(1:(maxIters + 1), benchmark_res[i,j], yaxis=:log, title = string("Nernst-Planck  ", strategies[i], minimizers[j]), ylabel = "log(loss)", legend = false)
            savefig(string(timePlot.title, ".pdf"))

            # Learning curve plots with different strategies, minimizer
            lossPlot = Plots.plot(1:(maxIters + 1), losses_res[i,j], yaxis=:log, title = string("Nernst-Planck  ", strategies[i], minimizers[j]), ylabel = "log(loss)", legend = false)
            savefig(string(lossPlot.title, ".pdf"))

            # Comparison between multiple strategies: maxIters fixed
            #   Here you can access the u_predic, u_numeric arrays of all the examples trained with different the strategies / minimzers.
            #   numeric_res[i,j] ...  prediction_res[i,j]
            # Make plots and comparisons
      end
end
