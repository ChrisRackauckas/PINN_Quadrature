#/////////////////////////////////////////////////////////////////////////////////
# INTERFACE TO RUN MUPLTIPLE EXAMPLES WITH DIFFERENT STRATEGIES / SETTINGS
#/////////////////////////////////////////////////////////////////////////////////
#using Plots

include("./allen_cahn_et.jl")

using JLD

# Settings:
maxIters = [(0,0,0,0,0,0,20000),(300,300,300,300,300,300,300)] #iters for ADAM/LBFGS

strategies = [NeuralPDE.QuadratureTraining(quadrature_alg = CubaCuhre(), reltol = 1, abstol = 1e-4, maxiters = 100),
              NeuralPDE.QuadratureTraining(quadrature_alg = HCubatureJL(), reltol = 1, abstol = 1e-4, maxiters = 100, batch = 0),
              NeuralPDE.QuadratureTraining(quadrature_alg = CubatureJLh(), reltol = 1, abstol = 1e-4, maxiters = 100),
              NeuralPDE.QuadratureTraining(quadrature_alg = CubatureJLp(), reltol = 1, abstol = 1e-4, maxiters = 100),
              NeuralPDE.GridTraining(0.09),
              NeuralPDE.StochasticTraining(100),
              NeuralPDE.QuasiRandomTraining(100; sampling_alg = UniformSample(), minibatch = 100)]

strategies_short_name = ["CubaCuhre",
                        "HCubatureJL",
                        "CubatureJLh",
                        "CubatureJLp",
                        "GridTraining",
                        "StochasticTraining",
                        "QuasiRandomTraining"]

minimizers = [GalacticOptim.ADAM(0.005)]
              #GalacticOptim.BFGS()]
              #GalacticOptim.LBFGS()]


minimizers_short_name = ["ADAM"]
                        #"LBFGS"]
                        # "BFGS"]


# Run models
error_res =  Dict()
domains = Dict()
params_res = Dict()  #to use same params for the next run
times = Dict()
losses_res = Dict()

print("Starting run")
## Convergence

for min =1:length(minimizers) # minimizer
      for strat=1:length(strategies) # strategy
            println(string(strategies_short_name[strat], "  ", minimizers_short_name[min]))
            res = allen_cahn(strategies[strat], minimizers[min], maxIters[min][strat])
            push!(error_res, string(strat,min)     => res[1])
            push!(params_res, string(strat,min) => res[2])
            push!(domains, string(strat,min)        => res[3])
            push!(times, string(strat,min)        => res[4])
            push!(losses_res, string(strat,min)        => res[5])
      end
end


save("./Allen_Cahn_Timeline.jld", "times", times)
save("./Allen_Cahn_Errors.jld", "error_res", error_res)
save("./Allen_Cahn_Params.jld", "params_res", params_res)
save("./Allen_Cahn_losses.jld", "losses_res", losses_res)



print("\n Plotting error vs iters")
#Plotting the first strategy with the first minimizer out from the loop to initialize the canvas
current_label = string(strategies_short_name[1], " + " , minimizers_short_name[1])
error = Plots.plot(times["11"], error_res["11"], yaxis=:log10, title = string("Level Set"), ylabel = "log(error)", label = current_label, xlims = (0,10))#legend = true)#, size=(1200,700))
plot!(error, times["21"], error_res["21"], yaxis=:log10, title = string("Level Set"), ylabel = "log(error)", label = string(strategies_short_name[2], " + " , minimizers_short_name[1]))
plot!(error, times["31"], error_res["31"], yaxis=:log10, title = string("Level Set"), ylabel = "log(error)", label = string(strategies_short_name[3], " + " , minimizers_short_name[1]))
plot!(error, times["41"], error_res["41"], yaxis=:log10, title = string("Level Set"), ylabel = "log(error)", label = string(strategies_short_name[4], " + " , minimizers_short_name[1]))
plot!(error, times["51"], error_res["51"], yaxis=:log10, title = string("Level Set"), ylabel = "log(error)", label = string(strategies_short_name[5], " + " , minimizers_short_name[1]))
plot!(error, times["61"], error_res["61"], yaxis=:log10, title = string("Level Set"), ylabel = "log(error)", label = string(strategies_short_name[6], " + " , minimizers_short_name[1]))
plot!(error, times["71"], error_res["71"], yaxis=:log10, title = string("Allen Cahn convergence"), ylabel = "log(error)", label = string(strategies_short_name[7], " + " , minimizers_short_name[1]))


plot!(error, times["12"], error_res["12"], yaxis=:log10, title = string("Level Set"), ylabel = "log(error)", label = string(strategies_short_name[1], " + " , minimizers_short_name[2]))
plot!(error, times["22"], error_res["22"], yaxis=:log10, title = string("Level Set"), ylabel = "log(error)", label = string(strategies_short_name[2], " + " , minimizers_short_name[2]))
plot!(error, times["32"], error_res["32"], yaxis=:log10, title = string("Level Set"), ylabel = "log(error)", label = string(strategies_short_name[3], " + " , minimizers_short_name[2]))
plot!(error, times["42"], error_res["42"], yaxis=:log10, title = string("Level Set"), ylabel = "log(error)", label = string(strategies_short_name[4], " + " , minimizers_short_name[2]))
plot!(error, times["52"], error_res["52"], yaxis=:log10, title = string("Level Set"), ylabel = "log(error)", label = string(strategies_short_name[5], " + " , minimizers_short_name[2]))
plot!(error, times["62"], error_res["62"], yaxis=:log10, title = string("Level Set"), ylabel = "log(error)", label = string(strategies_short_name[6], " + " , minimizers_short_name[2]))
plot!(error, times["72"], error_res["72"], yaxis=:log10, title = string("Allen Cahn convergence ADAM/LBFGS"), ylabel = "log(error)", label = string(strategies_short_name[7], " + " , minimizers_short_name[2]))


Plots.savefig("Allen_Cahn_error_vs_time.pdf")
