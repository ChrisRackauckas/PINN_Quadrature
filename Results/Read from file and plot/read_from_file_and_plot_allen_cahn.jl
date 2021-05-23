#/////////////////////////////////////////////////////////////////////////////////
# INTERFACE TO READ DATA FROM HPC RUNS
#/////////////////////////////////////////////////////////////////////////////////
using Plots
using JLD
using DelimitedFiles

strategies_short_name = ["CubaCuhre",
                        "HCubatureJL",
                        "CubatureJLh",
                        "CubatureJLp",
                        "GridTraining",
                        "StochasticTraining",
                        "QuasiRandomTraining"]

minimizers_short_name = ["ADAM",
                        # "BFGS",
                        "L-BFGS"]

#Read data
benchmark_res = Dict()
error_res =  Dict()
domains = Dict()

error_ac_res = load("/Users/francescocalisto/Documents/FRANCESCO/ACADEMICS/Università/MLJC/Sci-ML Julia/PINN_Quadrature-local/Results/run3_/Allen_Cahn_Errors.jld")["error_res"]
times_ac = load("/Users/francescocalisto/Documents/FRANCESCO/ACADEMICS/Università/MLJC/Sci-ML Julia/PINN_Quadrature-local/Results/run3_/Allen_Cahn_Timeline.jld")["times"]
pars_ac_res = load("/Users/francescocalisto/Documents/FRANCESCO/ACADEMICS/Università/MLJC/Sci-ML Julia/PINN_Quadrature-local/Results/run3_/Allen_Cahn_Params.jld")["params_res"]

error_ac_res
times_ac
pars_ac_res

times_ac_short = Dict()
error_ac_res_short = Dict()

for strat in 1:7
      times_ac[string(strat, "1")] = times_ac[string(strat, "1")].-times_ac[string(strat, "1")][1]
      times_ac_short[string(strat, "1")] = times_ac[string(strat, "1")][times_ac[string(strat, "1")] .< 2000]
      error_ac_res_short[string(strat, "1")] = error_ac_res[string(strat, "1")][1:length(times_ac_short[string(strat, "1")])]

      times_ac[string(strat, "2")] = times_ac[string(strat, "2")].-times_ac[string(strat, "2")][1]
      times_ac_short[string(strat, "2")] = times_ac[string(strat, "2")][times_ac[string(strat, "2")] .< 2000]
      error_ac_res_short[string(strat, "2")] = error_ac_res[string(strat, "2")][1:length(times_ac_short[string(strat, "2")])]
end


#Useful information
total_time = Dict()
final_error = Dict()
error_fixed_time = Dict()

for strat in 1:7
      for min in 1:2
            total_time[string(strat, "1")] = times_ac[string(strat, "1")][1000]
            final_error[string(strat, min)] = error_ac_res[string(strat, min)][end]
            error_fixed_time[string(strat, min)] = error_ac_res_short[string(strat, min)][end]
      end
end

total_time
final_error
error_fixed_time

benchmark_res_name = Dict()
for strat=1:length(strategies_short_name) # strategy
      for min =1:1#length(minimizers_short_name)
            push!(benchmark_res_name, string(strategies_short_name[strat]) => total_time[string(strat,min)])
      end
end

benchmark_res_name2 = Dict()
for strat=1:length(strategies_short_name) # strategy
      for min =2:2
            push!(benchmark_res_name2, string(strategies_short_name[strat]) => total_time[string(strat,min)])
      end
end

print("\n Plotting time benchmark")
#Plot benchmark ADAM
bar = Plots.bar(collect(keys(benchmark_res_name)), collect(values(benchmark_res_name)),
      title = string("Allen Cahn training time - ADAM(0.005) / 1k iters"), titlefontsize = 12, yrotation = 30, orientation= :horizontal,
      legend = false, xlabel = "Run time in seconds", xguidefontsize=9, size = (600,400))


#Plot benchmark L-BFGS
bar2 = Plots.bar(collect(keys(benchmark_res_name2)), collect(values(benchmark_res_name2)),
      title = string("Allen Cahn training time - L-BFGS"), titlefontsize = 12, yrotation = 30, orientation= :horizontal,
      legend = false, xlabel = "Run time in seconds", xguidefontsize=9, size = (600,400))


print("\n Plotting error vs iters")

for strat in 1:7
      for min in 1:1
            error_ac_res[string(strat, "1")] = error_ac_res[string(strat, "1")][1:1000]
      end
end
#Plot error vs iters ADAM
#Plotting the first strategy with the first minimizer out from the loop to initialize the canvas
current_label = string(strategies_short_name[1])
error = Plots.plot(1:length(error_ac_res["11"]), error_ac_res["11"], yaxis=:log10, title = string("Allen Cahn convergence - ADAM(0.005) / 1k iter."), ylabel = "Error", label = current_label)#legend = true)#, size=(1200,700))
plot!(error, 1:length(error_ac_res["21"]), error_ac_res[string(2,1)], yaxis=:log10, label = string(strategies_short_name[2]))
plot!(error, 1:length(error_ac_res["31"]), error_ac_res[string(3,1)], yaxis=:log10, label = string(strategies_short_name[3]))
plot!(error, 1:length(error_ac_res["41"]), error_ac_res[string(4,1)], yaxis=:log10, label = string(strategies_short_name[4]))
plot!(error, 1:length(error_ac_res["51"]), error_ac_res[string(5,1)], yaxis=:log10, label = string(strategies_short_name[5]))
plot!(error, 1:length(error_ac_res["61"]), error_ac_res[string(6,1)], yaxis=:log10, label = string(strategies_short_name[6]))
plot!(error, 1:length(error_ac_res["71"]), error_ac_res[string(7,1)], yaxis=:log10, label = string(strategies_short_name[7]))


Plots.plot(error, bar, layout = Plots.grid(1, 2, widths=[0.6 ,0.4]), size = (1500,500))

Plots.savefig("./Results/Error vs Iters/Allen Cahn/Allen_Cahn_comparison_1k_iters_ADAM.pdf")

#Plot error vs iters LBFGS
current_label = string(strategies_short_name[2])
error2 = Plots.plot(1:length(error_ac_res["11"]), error_ac_res["12"], yaxis=:log10, title = string("Allen Cahn convergence - L-BFGS"), ylabel = "Error", label = current_label)#legend = true)#, size=(1200,700))
plot!(error2, 1:length(error_ac_res["22"]), error_ac_res[string(2,2)], yaxis=:log10, label = string(strategies_short_name[2]))
plot!(error2, 1:length(error_ac_res["32"]), error_ac_res[string(3,2)], yaxis=:log10, label = string(strategies_short_name[3]))
plot!(error2, 1:length(error_ac_res["42"]), error_ac_res[string(4,2)], yaxis=:log10, label = string(strategies_short_name[4]))
plot!(error2, 1:length(error_ac_res["52"]), error_ac_res[string(5,2)], yaxis=:log10, label = string(strategies_short_name[5]))
plot!(error2, 1:length(error_ac_res["62"]), error_ac_res[string(6,2)], yaxis=:log10, label = string(strategies_short_name[6]))
plot!(error2, 1:length(error_ac_res["72"]), error_ac_res[string(7,2)], yaxis=:log10, label = string(strategies_short_name[7]))


Plots.plot(error2, bar2, layout = Plots.grid(1, 2, widths=[0.6 ,0.4]), size = (1500,500))

Plots.savefig("./Results/Error vs Iters/Allen Cahn/Allen_Cahn_comparison_LBFGS.pdf")



##PLOTS ERROR VS TIME
print("\n Plotting error_ac vs time")
#Plotting the first strategy with the first minimizer out from the loop to initialize the canvas
current_label = string(strategies_short_name[1])#, " + " , minimizers_short_name[1])
error_ac_adam = Plots.plot(times_ac_short["11"], error_ac_res_short["11"], yaxis=:log10, title = string("Allen Cahn convergence - ADAM (0.005)"), ylabel = "Error", label = current_label, xlabel = "Time (seconds)", size = (1500,500))#legend = true)#, size=(1200,700))
plot!(error_ac_adam, times_ac_short["21"], error_ac_res_short["21"], yaxis=:log10, label = string(strategies_short_name[2]))
plot!(error_ac_adam, times_ac_short["31"], error_ac_res_short["31"], yaxis=:log10, label = string(strategies_short_name[3]))#, " + " , minimizers_short_name[1]))
plot!(error_ac_adam, times_ac_short["41"], error_ac_res_short["41"], yaxis=:log10, label = string(strategies_short_name[4]))#, " + " , minimizers_short_name[1]))
plot!(error_ac_adam, times_ac_short["51"], error_ac_res_short["51"], yaxis=:log10, label = string(strategies_short_name[5]))#, " + " , minimizers_short_name[1]))
plot!(error_ac_adam, times_ac_short["61"], error_ac_res_short["61"], yaxis=:log10, label = string(strategies_short_name[6]))#, " + " , minimizers_short_name[1]))
plot!(error_ac_adam, times_ac_short["71"], error_ac_res_short["71"], yaxis=:log10, label = string(strategies_short_name[7]))#, " + " , minimizers_short_name[1]))

Plots.savefig("Allen_Cahn_et_ADAM_large.pdf")


#Plotting error vs time with L-BFGS
error_ac_lbfgs = Plots.plot(times_ac_short["12"], error_ac_res_short["12"], yaxis=:log10, title = string("Allen Cahn convergence - L-BFGS"), ylabel = "Error", label = string(strategies_short_name[1]), xlabel = "Time (seconds)")#, " + " , minimizers_short_name[2]))
plot!(error_ac_lbfgs, times_ac_short["22"], error_ac_res_short["22"], yaxis=:log10, label = string(strategies_short_name[2]))#, " + " , minimizers_short_name[2]))
plot!(error_ac_lbfgs, times_ac_short["32"], error_ac_res_short["32"], yaxis=:log10, label = string(strategies_short_name[3]))#, " + " , minimizers_short_name[2]))
plot!(error_ac_lbfgs, times_ac_short["42"], error_ac_res_short["42"], yaxis=:log10, label = string(strategies_short_name[4]))#, " + " , minimizers_short_name[2]))
plot!(error_ac_lbfgs, times_ac_short["52"], error_ac_res_short["52"], yaxis=:log10, label = string(strategies_short_name[5]))#, " + " , minimizers_short_name[2]))
plot!(error_ac_lbfgs, times_ac_short["62"], error_ac_res_short["62"], yaxis=:log10, label = string(strategies_short_name[6]))#, " + " , minimizers_short_name[2]))
plot!(error_ac_lbfgs, times_ac_short["72"], error_ac_res_short["72"], yaxis=:log10, label = string(strategies_short_name[7]))#, " + " , minimizers_short_name[2]))


Plots.savefig("Allen_Cahn_et_LBFGS_50s.pdf")
