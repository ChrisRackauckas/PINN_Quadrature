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

error_ls_res = load("/Users/francescocalisto/Documents/FRANCESCO/ACADEMICS/Università/MLJC/Sci-ML Julia/PINN_Quadrature-local/Results/Run4/Level_Set_Errors.jld")["error_res"]
times_ls = load("/Users/francescocalisto/Documents/FRANCESCO/ACADEMICS/Università/MLJC/Sci-ML Julia/PINN_Quadrature-local/Results/Run4/Level_Set_Timeline.jld")["times"]
pars_ls_res = load("/Users/francescocalisto/Documents/FRANCESCO/ACADEMICS/Università/MLJC/Sci-ML Julia/PINN_Quadrature-local/Results/Run4/Level_Set_Params.jld")["params_res"]

error_ls_res
times_ls
pars_ls_res

times_ls_short = Dict()
error_ls_res_short = Dict()

for strat in 1:7
      times_ls[string(strat, "1")] = times_ls[string(strat, "1")].-times_ls[string(strat, "1")][1]
      times_ls_short[string(strat, "1")] = times_ls[string(strat, "1")][times_ls[string(strat, "1")] .< 125]
      error_ls_res_short[string(strat, "1")] = error_ls_res[string(strat, "1")][1:length(times_ls_short[string(strat, "1")])]

      times_ls[string(strat, "2")] = times_ls[string(strat, "2")].-times_ls[string(strat, "2")][1]
      times_ls_short[string(strat, "2")] = times_ls[string(strat, "2")][times_ls[string(strat, "2")] .< 125]
      error_ls_res_short[string(strat, "2")] = error_ls_res[string(strat, "2")][1:length(times_ls_short[string(strat, "2")])]
end


#Useful information
total_time = Dict()
final_error = Dict()
min_error = Dict()
error_fixed_time = Dict()

for strat in 1:7
      for min in 1:2
            total_time[string(strat, min)] = times_ls[string(strat, min)][end]
            final_error[string(strat, min)] = error_ls_res[string(strat, min)][end]
            min_error[string(strat, min)] = minimum(error_ls_res[string(strat, min)])
            error_fixed_time[string(strat, min)] = error_ls_res_short[string(strat, min)][end]
      end
end

total_time
final_error
min_error
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
      title = string("Level Set training time - ADAM(0.005) / 20k iter."), titlefontsize = 12, yrotation = 30, orientation= :horizontal,
      legend = false, xlabel = "Run time in seconds", xguidefontsize=9, size = (600,400))


#Plot benchmark L-BFGS
bar2 = Plots.bar(collect(keys(benchmark_res_name2)), collect(values(benchmark_res_name2)),
      title = string("Level Set training time - L-BFGS"), titlefontsize = 12, yrotation = 30, orientation= :horizontal,
      legend = false, xlabel = "Run time in seconds", xguidefontsize=9, size = (600,400))


print("\n Plotting error vs iters")
#Plot error vs iters ADAM
#Plotting the first strategy with the first minimizer out from the loop to initialize the canvas
current_label = string(strategies_short_name[1])
error = Plots.plot(1:length(error_ls_res["11"]), error_ls_res["11"], yaxis=:log10, title = string("Level Set convergence - ADAM(0.005) / 20k iter."), ylabel = "Error", label = current_label)#legend = true)#, size=(1200,700))
plot!(error, 1:length(error_ls_res["21"]), error_ls_res[string(2,1)], yaxis=:log10, label = string(strategies_short_name[2]))
plot!(error, 1:length(error_ls_res["31"]), error_ls_res[string(3,1)], yaxis=:log10, label = string(strategies_short_name[3]))
plot!(error, 1:length(error_ls_res["41"]), error_ls_res[string(4,1)], yaxis=:log10, label = string(strategies_short_name[4]))
plot!(error, 1:length(error_ls_res["51"]), error_ls_res[string(5,1)], yaxis=:log10, label = string(strategies_short_name[5]))
plot!(error, 1:length(error_ls_res["61"]), error_ls_res[string(6,1)], yaxis=:log10, label = string(strategies_short_name[6]))
plot!(error, 1:length(error_ls_res["71"]), error_ls_res[string(7,1)], yaxis=:log10, label = string(strategies_short_name[7]))


Plots.plot(error, bar, layout = Plots.grid(1, 2, widths=[0.6 ,0.4]), size = (1500,500))

Plots.savefig("./Results/Plots/Error vs Iters/Level Set/Level_Set_comparison_20k_iters_ADAM_reltol_1.pdf")

#Plot error vs iters LBFGS
current_label = string(strategies_short_name[1])
error2 = Plots.plot(1:length(error_ls_res["11"]), error_ls_res["12"], yaxis=:log10, title = string("Level Set convergence - L-BFGS"), ylabel = "Error", label = current_label)#legend = true)#, size=(1200,700))
plot!(error2, 1:length(error_ls_res["22"]), error_ls_res[string(2,2)], yaxis=:log10, label = string(strategies_short_name[2]))
plot!(error2, 1:length(error_ls_res["32"]), error_ls_res[string(3,2)], yaxis=:log10, label = string(strategies_short_name[3]))
plot!(error2, 1:length(error_ls_res["42"]), error_ls_res[string(4,2)], yaxis=:log10, label = string(strategies_short_name[4]))
plot!(error2, 1:length(error_ls_res["52"]), error_ls_res[string(5,2)], yaxis=:log10, label = string(strategies_short_name[5]))
plot!(error2, 1:length(error_ls_res["62"]), error_ls_res[string(6,2)], yaxis=:log10, label = string(strategies_short_name[6]))
plot!(error2, 1:length(error_ls_res["72"]), error_ls_res[string(7,2)], yaxis=:log10, label = string(strategies_short_name[7]))


Plots.plot(error2, bar2, layout = Plots.grid(1, 2, widths=[0.6 ,0.4]), size = (1500,500))

Plots.savefig("./Results/Plots/Error vs Iters/Level Set/Level_Set_comparison_LBFGS_reltol_1.pdf")



##PLOTS ERROR VS TIME
print("\n Plotting error_ls vs time")
#Plotting the first strategy with the first minimizer out from the loop to initialize the canvas
current_label = string(strategies_short_name[1])#, " + " , minimizers_short_name[1])
error_ls_adam = Plots.plot(times_ls_short["11"], error_ls_res_short["11"], yaxis=:log10, title = string("Level Set convergence - ADAM (0.005) / First 125 seconds of training"), ylabel = "Error", label = current_label, xlabel = "Time (seconds)", size = (1500,500))#legend = true)#, size=(1200,700))
plot!(error_ls_adam, times_ls_short["21"], error_ls_res_short["21"], yaxis=:log10, label = string(strategies_short_name[2]))
plot!(error_ls_adam, times_ls_short["31"], error_ls_res_short["31"], yaxis=:log10, label = string(strategies_short_name[3]))#, " + " , minimizers_short_name[1]))
plot!(error_ls_adam, times_ls_short["41"], error_ls_res_short["41"], yaxis=:log10, label = string(strategies_short_name[4]))#, " + " , minimizers_short_name[1]))
plot!(error_ls_adam, times_ls_short["51"], error_ls_res_short["51"], yaxis=:log10, label = string(strategies_short_name[5]))#, " + " , minimizers_short_name[1]))
plot!(error_ls_adam, times_ls_short["61"], error_ls_res_short["61"], yaxis=:log10, label = string(strategies_short_name[6]))#, " + " , minimizers_short_name[1]))
plot!(error_ls_adam, times_ls_short["71"], error_ls_res_short["71"], yaxis=:log10, label = string(strategies_short_name[7]))#, " + " , minimizers_short_name[1]))

Plots.savefig("./Results/Plots/Error vs Time/Level Set/Level_Set_et_ADAM_125s_reltol_1.pdf")


#Plotting error vs time with L-BFGS
error_ls_lbfgs = Plots.plot(times_ls_short["12"], error_ls_res_short["12"], yaxis=:log10, title = string("Level Set convergence - L-BFGS"), ylabel = "Error", label = string(strategies_short_name[1]), xlabel = "Time (seconds)", size = (1500,500))#, " + " , minimizers_short_name[2]))
plot!(error_ls_lbfgs, times_ls_short["22"], error_ls_res_short["22"], yaxis=:log10, label = string(strategies_short_name[2]))#, " + " , minimizers_short_name[2]))
plot!(error_ls_lbfgs, times_ls_short["32"], error_ls_res_short["32"], yaxis=:log10, label = string(strategies_short_name[3]))#, " + " , minimizers_short_name[2]))
plot!(error_ls_lbfgs, times_ls_short["42"], error_ls_res_short["42"], yaxis=:log10, label = string(strategies_short_name[4]))#, " + " , minimizers_short_name[2]))
plot!(error_ls_lbfgs, times_ls_short["52"], error_ls_res_short["52"], yaxis=:log10, label = string(strategies_short_name[5]))#, " + " , minimizers_short_name[2]))
plot!(error_ls_lbfgs, times_ls_short["62"], error_ls_res_short["62"], yaxis=:log10, label = string(strategies_short_name[6]))#, " + " , minimizers_short_name[2]))
plot!(error_ls_lbfgs, times_ls_short["72"], error_ls_res_short["72"], yaxis=:log10, label = string(strategies_short_name[7]))#, " + " , minimizers_short_name[2]))


Plots.savefig("./Results/Plots/Error vs Time/Level Set/Level_Set_et_LBFGS_125s_reltol_1.pdf")



##ADAM vs L-BFGS comparison - first 200 iters
error_ls_res_short_iters = Dict()

for strat in 1:5
      for min in 1:2
            error_ls_res_short_iters[string(strat, min)] = error_ls_res[string(strat, min)][1:200]
      end
end
for strat in 6:7
      for min in 1:1
            error_ls_res_short_iters[string(strat, min)] = error_ls_res[string(strat, min)][1:200]
      end
end
for strat in 6:7
      for min in 2:2
            error_ls_res_short_iters[string(strat, min)] = error_ls_res[string(strat, min)]
      end
end


current_label = string(strategies_short_name[1])
error_adam = Plots.plot(1:length(error_ls_res_short_iters["11"]), error_ls_res_short_iters["11"], yaxis=:log10, title = string("Level Set convergence - ADAM(0.005) / First 200 iter."), ylabel = "Error", label = current_label, ylims = (0.0005,5))#legend = true)#, size=(1200,700))
plot!(error_adam, 1:length(error_ls_res_short_iters["21"]), error_ls_res_short_iters[string(2,1)], yaxis=:log10, label = string(strategies_short_name[2]))
plot!(error_adam, 1:length(error_ls_res_short_iters["31"]), error_ls_res_short_iters[string(3,1)], yaxis=:log10, label = string(strategies_short_name[3]))
plot!(error_adam, 1:length(error_ls_res_short_iters["41"]), error_ls_res_short_iters[string(4,1)], yaxis=:log10, label = string(strategies_short_name[4]))
plot!(error_adam, 1:length(error_ls_res_short_iters["51"]), error_ls_res_short_iters[string(5,1)], yaxis=:log10, label = string(strategies_short_name[5]))
plot!(error_adam, 1:length(error_ls_res_short_iters["61"]), error_ls_res_short_iters[string(6,1)], yaxis=:log10, label = string(strategies_short_name[6]))
plot!(error_adam, 1:length(error_ls_res_short_iters["71"]), error_ls_res_short_iters[string(7,1)], yaxis=:log10, label = string(strategies_short_name[7]))


#LBFGS
current_label = string(strategies_short_name[1])
error_lbfgs = Plots.plot(1:length(error_ls_res_short_iters["11"]), error_ls_res_short_iters["12"], yaxis=:log10, title = string("Level Set convergence - L-BFGS / First 200 iter."), ylabel = "Error", label = current_label, ylims = (0.0005,5))#legend = true)#, size=(1200,700))
plot!(error_lbfgs, 1:length(error_ls_res_short_iters["22"]), error_ls_res_short_iters[string(2,2)], yaxis=:log10, label = string(strategies_short_name[2]))
plot!(error_lbfgs, 1:length(error_ls_res_short_iters["32"]), error_ls_res_short_iters[string(3,2)], yaxis=:log10, label = string(strategies_short_name[3]))
plot!(error_lbfgs, 1:length(error_ls_res_short_iters["42"]), error_ls_res_short_iters[string(4,2)], yaxis=:log10, label = string(strategies_short_name[4]))
plot!(error_lbfgs, 1:length(error_ls_res_short_iters["52"]), error_ls_res_short_iters[string(5,2)], yaxis=:log10, label = string(strategies_short_name[5]))
plot!(error_lbfgs, 1:length(error_ls_res_short_iters["62"]), error_ls_res_short_iters[string(6,2)], yaxis=:log10, label = string(strategies_short_name[6]))
plot!(error_lbfgs, 1:length(error_ls_res_short_iters["72"]), error_ls_res_short_iters[string(7,2)], yaxis=:log10, label = string(strategies_short_name[7]))

Plots.plot(error_adam, error_lbfgs, layout = Plots.grid(1, 2, widths=[0.5 ,0.5]), size = (1500,500))
Plots.savefig("./Results/Plots/Error vs Iters/Level Set/Level_Set_ADAM_vs_LBFGS_100_iter_reltol_1.pdf")
