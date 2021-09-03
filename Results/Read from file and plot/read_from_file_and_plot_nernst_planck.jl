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

error_np_res = load("/Users/francescocalisto/Documents/FRANCESCO/ACADEMICS/Università/MLJC/Sci-ML Julia/PINN_Quadrature-local/Results/run3_/nernst_error.jld")["error_res"]
times_np = load("/Users/francescocalisto/Documents/FRANCESCO/ACADEMICS/Università/MLJC/Sci-ML Julia/PINN_Quadrature-local/Results/run3_/nernst_times.jld")["times"]
pars_np_res = load("/Users/francescocalisto/Documents/FRANCESCO/ACADEMICS/Università/MLJC/Sci-ML Julia/PINN_Quadrature-local/Results/run3_/nernst_params.jld")["params_res"]

error_np_res
times_np
pars_np_res

error_np_res["72"] = [0]
times_np["72"] = [0]
pars_np_res["72"] = [0]

times_np_short = Dict()
error_np_res_short = Dict()

for strat in 1:7
      times_np[string(strat, "1")] = times_np[string(strat, "1")].-times_np[string(strat, "1")][1]
      times_np_short[string(strat, "1")] = times_np[string(strat, "1")][times_np[string(strat, "1")] .< 400]
      error_np_res_short[string(strat, "1")] = error_np_res[string(strat, "1")][1:length(times_np_short[string(strat, "1")])]

      times_np[string(strat, "2")] = times_np[string(strat, "2")].-times_np[string(strat, "2")][1]
      times_np_short[string(strat, "2")] = times_np[string(strat, "2")][times_np[string(strat, "2")] .< 2000]
      error_np_res_short[string(strat, "2")] = error_np_res[string(strat, "2")][1:length(times_np_short[string(strat, "2")])]
end


#Useful information
total_time = Dict()
final_error = Dict()
error_fixed_time = Dict()

for strat in 1:7
      for min in 1:2
            total_time[string(strat, "1")] = times_np[string(strat, "1")][10000]
            final_error[string(strat, min)] = error_np_res[string(strat, min)][end]
            error_fixed_time[string(strat, min)] = error_np_res_short[string(strat, min)][end]
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
      title = string("Nernst Planck training time - ADAM(0.005) / 10k iters"), titlefontsize = 12, yrotation = 30, orientation= :horizontal,
      legend = false, xlabel = "Run time in seconds", xguidefontsize=9, size = (600,400))


#Plot benchmark L-BFGS
bar2 = Plots.bar(collect(keys(benchmark_res_name2)), collect(values(benchmark_res_name2)),
      title = string("Nernst Planck training time - L-BFGS"), titlefontsize = 12, yrotation = 30, orientation= :horizontal,
      legend = false, xlabel = "Run time in seconds", xguidefontsize=9, size = (600,400))


print("\n Plotting error vs iters")

for strat in 1:7
      for min in 1:1
            error_np_res[string(strat, "1")] = error_np_res[string(strat, "1")][1:10000]
      end
end
#Plot error vs iters ADAM
#Plotting the first strategy with the first minimizer out from the loop to initialize the canvas
current_label = string(strategies_short_name[1])
error = Plots.plot(1:length(error_np_res["11"]), error_np_res["11"], yaxis=:log10, title = string("Nernst Planck convergence - ADAM(0.005) / 10k iter."), ylabel = "Error", label = current_label)#legend = true)#, size=(1200,700))
plot!(error, 1:length(error_np_res["21"]), error_np_res[string(2,1)], yaxis=:log10, label = string(strategies_short_name[2]))
plot!(error, 1:length(error_np_res["31"]), error_np_res[string(3,1)], yaxis=:log10, label = string(strategies_short_name[3]))
plot!(error, 1:length(error_np_res["41"]), error_np_res[string(4,1)], yaxis=:log10, label = string(strategies_short_name[4]))
plot!(error, 1:length(error_np_res["51"]), error_np_res[string(5,1)], yaxis=:log10, label = string(strategies_short_name[5]))
plot!(error, 1:length(error_np_res["61"]), error_np_res[string(6,1)], yaxis=:log10, label = string(strategies_short_name[6]))
plot!(error, 1:length(error_np_res["71"]), error_np_res[string(7,1)], yaxis=:log10, label = string(strategies_short_name[7]))


Plots.plot(error, bar, layout = Plots.grid(1, 2, widths=[0.6 ,0.4]), size = (1500,500))

Plots.savefig("./Results/Error vs Iters/Nernst Planck/Nernst_Planck_comparison_10k_iters_ADAM.pdf")

#Plot error vs iters LBFGS
current_label = string(strategies_short_name[2])
error2 = Plots.plot(1:length(error_np_res["11"]), error_np_res["12"], yaxis=:log10, title = string("Nernst Planck convergence - L-BFGS"), ylabel = "Error", label = current_label)#legend = true)#, size=(1200,700))
plot!(error2, 1:length(error_np_res["22"]), error_np_res[string(2,2)], yaxis=:log10, label = string(strategies_short_name[2]))
plot!(error2, 1:length(error_np_res["32"]), error_np_res[string(3,2)], yaxis=:log10, label = string(strategies_short_name[3]))
plot!(error2, 1:length(error_np_res["42"]), error_np_res[string(4,2)], yaxis=:log10, label = string(strategies_short_name[4]))
plot!(error2, 1:length(error_np_res["52"]), error_np_res[string(5,2)], yaxis=:log10, label = string(strategies_short_name[5]))
plot!(error2, 1:length(error_np_res["62"]), error_np_res[string(6,2)], yaxis=:log10, label = string(strategies_short_name[6]))
plot!(error2, 1:length(error_np_res["72"]), error_np_res[string(7,2)], yaxis=:log10, label = string(strategies_short_name[7]))


Plots.plot(error2, bar2, layout = Plots.grid(1, 2, widths=[0.6 ,0.4]), size = (1500,500))

Plots.savefig("./Results/Error vs Iters/Nernst Planck/Nernst_Planck_comparison_LBFGS.pdf")



##PLOTS ERROR VS TIME
print("\n Plotting error_np vs time")
#Plotting the first strategy with the first minimizer out from the loop to initialize the canvas
current_label = string(strategies_short_name[1])#, " + " , minimizers_short_name[1])
error_np_adam = Plots.plot(times_np_short["11"], error_np_res_short["11"], yaxis=:log10, title = string("Nernst Planck convergence - ADAM (0.005) / First 400 seconds of training"), ylabel = "Error", label = current_label, xlabel = "Time (seconds)", size = (1500,500))#legend = true)#, size=(1200,700))
plot!(error_np_adam, times_np_short["21"], error_np_res_short["21"], yaxis=:log10, label = string(strategies_short_name[2]))
plot!(error_np_adam, times_np_short["31"], error_np_res_short["31"], yaxis=:log10, label = string(strategies_short_name[3]))#, " + " , minimizers_short_name[1]))
plot!(error_np_adam, times_np_short["41"], error_np_res_short["41"], yaxis=:log10, label = string(strategies_short_name[4]))#, " + " , minimizers_short_name[1]))
plot!(error_np_adam, times_np_short["51"], error_np_res_short["51"], yaxis=:log10, label = string(strategies_short_name[5]))#, " + " , minimizers_short_name[1]))
plot!(error_np_adam, times_np_short["61"], error_np_res_short["61"], yaxis=:log10, label = string(strategies_short_name[6]))#, " + " , minimizers_short_name[1]))
plot!(error_np_adam, times_np_short["71"], error_np_res_short["71"], yaxis=:log10, label = string(strategies_short_name[7]))#, " + " , minimizers_short_name[1]))

Plots.savefig("./Results/Error vs Time/Nernst Planck/Nernst_Planck_et_ADAM_large_400s.pdf")


#Plotting error vs time with L-BFGS
error_np_lbfgs = Plots.plot(times_np_short["12"], error_np_res_short["12"], yaxis=:log10, title = string("Nernst Planck convergence - L-BFGS"), ylabel = "Error", label = string(strategies_short_name[1]), xlabel = "Time (seconds)")#, " + " , minimizers_short_name[2]))
plot!(error_np_lbfgs, times_np_short["22"], error_np_res_short["22"], yaxis=:log10, label = string(strategies_short_name[2]))#, " + " , minimizers_short_name[2]))
plot!(error_np_lbfgs, times_np_short["32"], error_np_res_short["32"], yaxis=:log10, label = string(strategies_short_name[3]))#, " + " , minimizers_short_name[2]))
plot!(error_np_lbfgs, times_np_short["42"], error_np_res_short["42"], yaxis=:log10, label = string(strategies_short_name[4]))#, " + " , minimizers_short_name[2]))
plot!(error_np_lbfgs, times_np_short["52"], error_np_res_short["52"], yaxis=:log10, label = string(strategies_short_name[5]))#, " + " , minimizers_short_name[2]))
plot!(error_np_lbfgs, times_np_short["62"], error_np_res_short["62"], yaxis=:log10, label = string(strategies_short_name[6]))#, " + " , minimizers_short_name[2]))
plot!(error_np_lbfgs, times_np_short["72"], error_np_res_short["72"], yaxis=:log10, label = string(strategies_short_name[7]))#, " + " , minimizers_short_name[2]))


Plots.savefig("Nernst_Planck_et_LBFGS_50s.pdf")
