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

error_hj_res = load("/Users/francescocalisto/Documents/FRANCESCO/ACADEMICS/Università/MLJC/Sci-ML Julia/PINN_Quadrature-local/Results/run2_/HLB_errors.jld")["error_res"]
times_hj = load("/Users/francescocalisto/Documents/FRANCESCO/ACADEMICS/Università/MLJC/Sci-ML Julia/PINN_Quadrature-local/Results/run2_/HLB_times.jld")["times"]
pars_hj_res = load("/Users/francescocalisto/Documents/FRANCESCO/ACADEMICS/Università/MLJC/Sci-ML Julia/PINN_Quadrature-local/Results/run2_/HLB_params.jld")["params_res"]

error_hj_res
times_hj
pars_hj_res

times_hj_short = Dict()
error_hj_res_short = Dict()

for strat in 1:5
      times_hj[string(strat, "1")] = times_hj[string(strat, "1")].-times_hj[string(strat, "1")][1]
      times_hj_short[string(strat, "1")] = times_hj[string(strat, "1")][times_hj[string(strat, "1")] .< 220]
      error_hj_res_short[string(strat, "1")] = error_hj_res[string(strat, "1")][1:length(times_hj_short[string(strat, "1")])]

      times_hj[string(strat, "2")] = times_hj[string(strat, "2")].-times_hj[string(strat, "2")][1]
      times_hj_short[string(strat, "2")] = times_hj[string(strat, "2")][times_hj[string(strat, "2")] .< 50]
      error_hj_res_short[string(strat, "2")] = error_hj_res[string(strat, "2")][1:length(times_hj_short[string(strat, "2")])]
end


#Useful information
total_time = Dict()
final_error = Dict()
error_fixed_time = Dict()

for strat in 1:5
      for min in 1:2
            total_time[string(strat, min)] = times_hj[string(strat, min)][end]
            final_error[string(strat, min)] = error_hj_res[string(strat, min)][end]
            error_fixed_time[string(strat, min)] = error_hj_res_short[string(strat, min)][end]
      end
end

total_time
final_error
error_fixed_time

benchmark_res_name = Dict()
for strat=1:5#length(strategies_short_name) # strategy
      for min =1:1#length(minimizers_short_name)
            push!(benchmark_res_name, string(strategies_short_name[strat]) => total_time[string(strat,min)])
      end
end

benchmark_res_name2 = Dict()
for strat=1:5#length(strategies_short_name) # strategy
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
error = Plots.plot(1:length(error_hj_res["11"]), error_hj_res["11"], yaxis=:log10, title = string("Level Set convergence - ADAM(0.005) / 20k iter."), ylabel = "Error", label = current_label)#legend = true)#, size=(1200,700))
plot!(error, 1:length(error_hj_res["21"]), error_hj_res[string(2,1)], yaxis=:log10, label = string(strategies_short_name[2]))
plot!(error, 1:length(error_hj_res["31"]), error_hj_res[string(3,1)], yaxis=:log10, label = string(strategies_short_name[3]))
plot!(error, 1:length(error_hj_res["41"]), error_hj_res[string(4,1)], yaxis=:log10, label = string(strategies_short_name[4]))
plot!(error, 1:length(error_hj_res["51"]), error_hj_res[string(5,1)], yaxis=:log10, label = string(strategies_short_name[5]))
#plot!(error, 1:length(error_hj_res["61"]), error_hj_res[string(6,1)][1:20001], yaxis=:log10, label = string(strategies_short_name[6]))
#plot!(error, 1:length(error_hj_res["71"]), error_hj_res[string(7,1)][1:20001], yaxis=:log10, label = string(strategies_short_name[7]))


Plots.plot(error, bar, layout = Plots.grid(1, 2, widths=[0.6 ,0.4]), size = (1500,500))

Plots.savefig("./Results/Error vs Iters/Level Set/Level_Set_comparison_20k_iters_ADAM.pdf")

#Plot error vs iters LBFGS
current_label = string(strategies_short_name[2])
error2 = Plots.plot(1:length(error_hj_res["11"]), error_hj_res["12"], yaxis=:log10, title = string("Level Set convergence - L-BFGS"), ylabel = "Error", label = current_label)#legend = true)#, size=(1200,700))
plot!(error2, 1:length(error_hj_res["22"]), error_hj_res[string(2,2)], yaxis=:log10, label = string(strategies_short_name[2]))
plot!(error2, 1:length(error_hj_res["32"]), error_hj_res[string(3,2)], yaxis=:log10, label = string(strategies_short_name[3]))
plot!(error2, 1:length(error_hj_res["42"]), error_hj_res[string(4,2)], yaxis=:log10, label = string(strategies_short_name[4]))
plot!(error2, 1:length(error_hj_res["52"]), error_hj_res[string(5,2)], yaxis=:log10, label = string(strategies_short_name[5]))
plot!(error2, 1:length(error_hj_res["62"]), error_hj_res[string(6,2)], yaxis=:log10, label = string(strategies_short_name[6]))
plot!(error2, 1:length(error_hj_res["72"]), error_hj_res[string(7,2)], yaxis=:log10, label = string(strategies_short_name[7]))


Plots.plot(error2, bar2, layout = Plots.grid(1, 2, widths=[0.6 ,0.4]), size = (1500,500))

Plots.savefig("./Results/Error vs Iters/Level Set/Level_Set_comparison_LBFGS.pdf")



##PLOTS ERROR VS TIME
print("\n Plotting error_hj vs time")
#Plotting the first strategy with the first minimizer out from the loop to initialize the canvas
current_label = string(strategies_short_name[1])#, " + " , minimizers_short_name[1])
error_hj_adam = Plots.plot(times_hj_short["11"], error_hj_res_short["11"], yaxis=:log10, title = string("Level Set convergence - ADAM (0.005)"), ylabel = "Error", label = current_label, xlabel = "Time (seconds)", size = (1500,500))#legend = true)#, size=(1200,700))
plot!(error_hj_adam, times_hj_short["21"], error_hj_res_short["21"], yaxis=:log10, label = string(strategies_short_name[2]))
plot!(error_hj_adam, times_hj_short["31"], error_hj_res_short["31"], yaxis=:log10, label = string(strategies_short_name[3]))#, " + " , minimizers_short_name[1]))
plot!(error_hj_adam, times_hj_short["41"], error_hj_res_short["41"], yaxis=:log10, label = string(strategies_short_name[4]))#, " + " , minimizers_short_name[1]))
plot!(error_hj_adam, times_hj_short["51"], error_hj_res_short["51"], yaxis=:log10, label = string(strategies_short_name[5]))#, " + " , minimizers_short_name[1]))
plot!(error_hj_adam, times_hj_short["61"], error_hj_res_short["61"], yaxis=:log10, label = string(strategies_short_name[6]))#, " + " , minimizers_short_name[1]))
plot!(error_hj_adam, times_hj_short["71"], error_hj_res_short["71"], yaxis=:log10, label = string(strategies_short_name[7]))#, " + " , minimizers_short_name[1]))

Plots.savefig("Level_Set_et_ADAM_large.pdf")


#Plotting error vs time with L-BFGS
error_hj_lbfgs = Plots.plot(times_hj_short["12"], error_hj_res_short["12"], yaxis=:log10, title = string("Level Set convergence - L-BFGS"), ylabel = "Error", label = string(strategies_short_name[1]), xlabel = "Time (seconds)")#, " + " , minimizers_short_name[2]))
plot!(error_hj_lbfgs, times_hj_short["22"], error_hj_res_short["22"], yaxis=:log10, label = string(strategies_short_name[2]))#, " + " , minimizers_short_name[2]))
plot!(error_hj_lbfgs, times_hj_short["32"], error_hj_res_short["32"], yaxis=:log10, label = string(strategies_short_name[3]))#, " + " , minimizers_short_name[2]))
plot!(error_hj_lbfgs, times_hj_short["42"], error_hj_res_short["42"], yaxis=:log10, label = string(strategies_short_name[4]))#, " + " , minimizers_short_name[2]))
plot!(error_hj_lbfgs, times_hj_short["52"], error_hj_res_short["52"], yaxis=:log10, label = string(strategies_short_name[5]))#, " + " , minimizers_short_name[2]))
plot!(error_hj_lbfgs, times_hj_short["62"], error_hj_res_short["62"], yaxis=:log10, label = string(strategies_short_name[6]))#, " + " , minimizers_short_name[2]))
plot!(error_hj_lbfgs, times_hj_short["72"], error_hj_res_short["72"], yaxis=:log10, label = string(strategies_short_name[7]))#, " + " , minimizers_short_name[2]))


Plots.savefig("Level_Set_et_LBFGS_50s.pdf")
