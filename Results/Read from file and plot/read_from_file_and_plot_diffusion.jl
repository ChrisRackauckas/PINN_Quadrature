#/////////////////////////////////////////////////////////////////////////////////
# INTERFACE TO READ DATA FROM HPC RUNS
#/////////////////////////////////////////////////////////////////////////////////
using Plots
using JLD
using DelimitedFiles
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature,Cubature, Cuba
using QuasiMonteCarlo
using Statistics

strategies = [#NeuralPDE.QuadratureTraining(quadrature_alg = CubaCuhre(), reltol = 1e-4, abstol = 1e-3, maxiters = 10, batch = 10),
              NeuralPDE.QuadratureTraining(quadrature_alg = HCubatureJL(), reltol=1, abstol=1e-5, maxiters=100, batch = 0),
              NeuralPDE.QuadratureTraining(quadrature_alg = CubatureJLh(), reltol=1, abstol=1e-5, maxiters=100),
              NeuralPDE.QuadratureTraining(quadrature_alg = CubatureJLp(), reltol=1, abstol=1e-5, maxiters=100),
              NeuralPDE.GridTraining([0.2,0.1]),
              NeuralPDE.StochasticTraining(100),
              NeuralPDE.QuasiRandomTraining(100; sampling_alg = UniformSample(), minibatch = 100)]


strategies_short_name = [#"CubaCuhre",
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
prediction_res =  Dict()
benchmark_res = Dict()
error_res =  Dict()
domains = Dict()

error_d_res_1 = load("/Users/francescocalisto/Documents/FRANCESCO/ACADEMICS/Università/MLJC/Sci-ML Julia/PINN_Quadrature-local/Results/Run4/diffusion_errors_run1.jld")["error_res"]
times_d_1 = load("/Users/francescocalisto/Documents/FRANCESCO/ACADEMICS/Università/MLJC/Sci-ML Julia/PINN_Quadrature-local/Results/Run4/diffusion_times_run1.jld")["times"]
pars_d_res_1 = load("/Users/francescocalisto/Documents/FRANCESCO/ACADEMICS/Università/MLJC/Sci-ML Julia/PINN_Quadrature-local/Results/Run4/diffusion_params_run1.jld")["params_res"]

error_d_res_2 = load("/Users/francescocalisto/Documents/FRANCESCO/ACADEMICS/Università/MLJC/Sci-ML Julia/PINN_Quadrature-local/Results/Run4/diffusion_Errors_run2.jld")["error_res"]
times_d_2 = load("/Users/francescocalisto/Documents/FRANCESCO/ACADEMICS/Università/MLJC/Sci-ML Julia/PINN_Quadrature-local/Results/Run4/diffusion_Timeline_run2.jld")["times"]
pars_d_res_2 = load("/Users/francescocalisto/Documents/FRANCESCO/ACADEMICS/Università/MLJC/Sci-ML Julia/PINN_Quadrature-local/Results/Run4/diffusion_Params_run2.jld")["params_res"]

error_d_res = Dict()
times_d = Dict()
pars_d_res = Dict()

for strat in 1:6
      error_d_res[string(strat,"1")] = error_d_res_1[string(strat,"1")]
      times_d[string(strat,"1")] = times_d_1[string(strat,"1")]
      pars_d_res[string(strat,"1")] = pars_d_res_1[string(strat,"1")]

      error_d_res[string(strat,"2")] = error_d_res_2[string(strat,"2")]
      times_d[string(strat,"2")] = times_d_2[string(strat,"2")]
      pars_d_res[string(strat,"2")] = pars_d_res_2[string(strat,"2")]
end

error_d_res
times_d
pars_d_res

times_d_short = Dict()
error_d_res_short = Dict()

for strat in 1:6
      times_d[string(strat, "1")] = times_d[string(strat, "1")].-times_d[string(strat, "1")][1]
      times_d_short[string(strat, "1")] = times_d[string(strat, "1")][times_d[string(strat, "1")] .< 500]
      error_d_res_short[string(strat, "1")] = error_d_res[string(strat, "1")][1:length(times_d_short[string(strat, "1")])]

      times_d[string(strat, "2")] = times_d[string(strat, "2")].-times_d[string(strat, "2")][1]
      times_d_short[string(strat, "2")] = times_d[string(strat, "2")][times_d[string(strat, "2")] .< 50]
      error_d_res_short[string(strat, "2")] = error_d_res[string(strat, "2")][1:length(times_d_short[string(strat, "2")])]
end


#Useful information
total_time = Dict()
final_error = Dict()
min_error = Dict()
error_fixed_time = Dict()

for strat in 1:6
      for min in 1:2
            total_time[string(strat, min)] = times_d[string(strat, min)][end]
            final_error[string(strat, min)] = error_d_res[string(strat, min)][end]
            min_error[string(strat, min)] = minimum(error_d_res[string(strat, min)])
            error_fixed_time[string(strat, min)] = error_d_res_short[string(strat, min)][end]
      end
end

total_time
final_error
min_error
error_fixed_time

benchmark_res_name = Dict()
for strat in 1:length(strategies_short_name) # strategy
      for min in 1:1#length(minimizers_short_name)
            push!(benchmark_res_name, string(strategies_short_name[strat]) => total_time[string(strat,min)])
      end
end

benchmark_res_name
delete!(benchmark_res_name, "CubatureJLp")
benchmark_res_name

benchmark_res_name2 = Dict()
for strat in 1:length(strategies_short_name) # strategy
      push!(benchmark_res_name2, string(strategies_short_name[strat]) => total_time[string(strat,"2")])
end

benchmark_res_name2
delete!(benchmark_res_name2, "CubatureJLp")
benchmark_res_name2

print("\n Plotting time benchmark")
#Plot benchmark ADAM
bar = Plots.bar(collect(keys(benchmark_res_name)), collect(values(benchmark_res_name)),
      title = string("Diffusion training time - ADAM(0.001) / 50k iter."), titlefontsize = 12, yrotation = 30, orientation= :horizontal,
      legend = false, xlabel = "Run time in seconds", xguidefontsize=9, size = (600,400))

#Plot benchmark L-BFGS
bar2 = Plots.bar(collect(keys(benchmark_res_name2)), collect(values(benchmark_res_name2)),
      title = string("Diffusion training time - L-BFGS"), titlefontsize = 12, yrotation = 30, orientation= :horizontal,
      legend = false, xlabel = "Run time in seconds", xguidefontsize=9, size = (600,400))


print("\n Plotting error vs iters")
#Plot error vs iters ADAM
#Plotting the first strategy with the first minimizer out from the loop to initialize the canvas
current_label = string(strategies_short_name[1])
error = Plots.plot(1:length(error_d_res["11"]), error_d_res["11"], yaxis=:log10, title = string("Diffusion convergence - ADAM(0.001) / 50k iter."), ylabel = "Error", label = current_label)
plot!(error, 1:length(error_d_res["21"]), error_d_res[string(2,1)], yaxis=:log10, label = string(strategies_short_name[2]))
#plot!(error, 1:length(error_d_res["31"]), error_d_res[string(3,1)], yaxis=:log10, label = string(strategies_short_name[3]))
plot!(error, 1:length(error_d_res["41"]), error_d_res[string(4,1)], yaxis=:log10, label = string(strategies_short_name[4]))
plot!(error, 1:length(error_d_res["51"]), error_d_res[string(5,1)], yaxis=:log10, label = string(strategies_short_name[5]))
plot!(error, 1:length(error_d_res["61"]), error_d_res[string(6,1)], yaxis=:log10, label = string(strategies_short_name[6]))


Plots.plot(error, bar, layout = Plots.grid(1, 2, widths=[0.6 ,0.4]), size = (1500,500))

Plots.savefig("./Results/Plots/Error vs Iters/Diffusion/diffusion_comparison_50k_iters_ADAM_1_reltol.pdf")

#Plot error vs iters LBFGS
current_label = string(strategies_short_name[1])
error2 = Plots.plot(1:length(error_d_res["12"]), error_d_res["12"], yaxis=:log10, title = string("Diffusion convergence - L-BFGS"), ylabel = "Error", label = current_label)#legend = true)#, size=(1200,700))
plot!(error2, 1:length(error_d_res["22"]), error_d_res[string(2,2)], yaxis=:log10, label = string(strategies_short_name[2]))
#plot!(error2, 1:600, error_d_res[string(3,2)][1:600], yaxis=:log10, label = string(strategies_short_name[3]))
plot!(error2, 1:length(error_d_res["42"]), error_d_res[string(4,2)], yaxis=:log10, label = string(strategies_short_name[4]))
plot!(error2, 1:length(error_d_res["52"]), error_d_res[string(5,2)], yaxis=:log10, label = string(strategies_short_name[5]))
plot!(error2, 1:length(error_d_res["62"]), error_d_res[string(6,2)], yaxis=:log10, label = string(strategies_short_name[6]))


Plots.plot(error2, bar2, layout = Plots.grid(1, 2, widths=[0.6 ,0.4]), size = (1500,500))

Plots.savefig("./Results/Plots/Error vs Iters/Diffusion/diffusion_comparison_LBFGS_reltol_1.pdf")


##PLOTS ERROR VS TIME
#Plotting error vs time with ADAM
print("\n Plotting error_d vs time")
#Plotting the first strategy with the first minimizer out from the loop to initialize the canvas
current_label = string(strategies_short_name[1])#, " + " , minimizers_short_name[1])
error_d_adam = Plots.plot(times_d_short["11"], error_d_res_short["11"], yaxis=:log10, title = string("Diffusion convergence - ADAM (0.001) / First 300 seconds of training"), ylabel = "Error", label = current_label, xlabel = "Time (seconds)", size = (1500,500))#legend = true)#, size=(1200,700))
plot!(error_d_adam, times_d_short["21"], error_d_res_short["21"], yaxis=:log10, label = string(strategies_short_name[2]))#, ylabel = "log(error_d)", label = string(strategies_short_name[2]))#, " + " , minimizers_short_name[1]))
#plot!(error_d_adam, times_d_short["31"], error_d_res_short["31"], yaxis=:log10, label = string(strategies_short_name[3]))#, " + " , minimizers_short_name[1]))
plot!(error_d_adam, times_d_short["41"], error_d_res_short["41"], yaxis=:log10, label = string(strategies_short_name[4]))#, " + " , minimizers_short_name[1]))
plot!(error_d_adam, times_d_short["51"], error_d_res_short["51"], yaxis=:log10, label = string(strategies_short_name[5]))#, " + " , minimizers_short_name[1]))
plot!(error_d_adam, times_d_short["61"], error_d_res_short["61"], yaxis=:log10, label = string(strategies_short_name[6]))#, " + " , minimizers_short_name[1]))

Plots.savefig("./Results/Plots/Error vs Time/Diffusion/Diffusion_et_ADAM_1_reltol.pdf")


#Plotting error vs time with L-BFGS
error_d_lbfgs = Plots.plot(times_d_short["12"], error_d_res_short["12"], yaxis=:log10, title = string("Diffusion convergence - L-BFGS"), ylabel = "Error", label = string(strategies_short_name[1]), xlabel = "Time (seconds)")
plot!(error_d_lbfgs, times_d_short["22"], error_d_res_short["22"], yaxis=:log10, label = string(strategies_short_name[2]))#, " + " , minimizers_short_name[2]))
#plot!(error_d_lbfgs, times_d_short["32"], error_d_res_short["32"], yaxis=:log10, label = string(strategies_short_name[3]))#, " + " , minimizers_short_name[2]))
plot!(error_d_lbfgs, times_d_short["42"], error_d_res_short["42"], yaxis=:log10, label = string(strategies_short_name[4]))#, " + " , minimizers_short_name[2]))
plot!(error_d_lbfgs, times_d_short["52"], error_d_res_short["52"], yaxis=:log10, label = string(strategies_short_name[5]))#, " + " , minimizers_short_name[2]))
plot!(error_d_lbfgs, times_d_short["62"], error_d_res_short["62"], yaxis=:log10, label = string(strategies_short_name[6]))#, " + " , minimizers_short_name[2]))

Plots.savefig("Diffusion_et_LBFGS.pdf")



##ADAM vs L-BFGS comparison - first 200 iters
error_d_res_short_iters = Dict()

for strat in 1:3
      for min in 1:2
            error_d_res_short_iters[string(strat, min)] = error_d_res[string(strat, min)][1:50]
      end
end
for strat in 4:6
      for min in 1:1
            error_d_res_short_iters[string(strat, min)] = error_d_res[string(strat, min)][1:50]
      end
end
for strat in 4:6
      for min in 2:2
            error_d_res_short_iters[string(strat, min)] = error_d_res[string(strat, min)]
      end
end


current_label = string(strategies_short_name[1])
error_adam = Plots.plot(1:length(error_d_res_short_iters["11"]), error_d_res_short_iters["11"], yaxis=:log10, title = string("Diffusion convergence - ADAM(0.005) / First 50 iter."), ylabel = "Error", label = current_label, ylims = (0.1,15))#legend = true)#, size=(1200,700))
plot!(error_adam, 1:length(error_d_res_short_iters["21"]), error_d_res_short_iters[string(2,1)], yaxis=:log10, label = string(strategies_short_name[2]))
#plot!(error_adam, 1:length(error_d_res_short_iters["31"]), error_d_res_short_iters[string(3,1)], yaxis=:log10, label = string(strategies_short_name[3]))
plot!(error_adam, 1:length(error_d_res_short_iters["41"]), error_d_res_short_iters[string(4,1)], yaxis=:log10, label = string(strategies_short_name[4]))
plot!(error_adam, 1:length(error_d_res_short_iters["51"]), error_d_res_short_iters[string(5,1)], yaxis=:log10, label = string(strategies_short_name[5]))
plot!(error_adam, 1:length(error_d_res_short_iters["61"]), error_d_res_short_iters[string(6,1)], yaxis=:log10, label = string(strategies_short_name[6]))

#LBFGS
current_label = string(strategies_short_name[1])
error_lbfgs = Plots.plot(1:length(error_d_res_short_iters["11"]), error_d_res_short_iters["12"], yaxis=:log10, title = string("Diffusion convergence - L-BFGS / First 50 iter."), ylabel = "Error", label = current_label, ylims = (0.1,15))#legend = true)#, size=(1200,700))
plot!(error_lbfgs, 1:length(error_d_res_short_iters["22"]), error_d_res_short_iters[string(2,2)], yaxis=:log10, label = string(strategies_short_name[2]))
#plot!(error_lbfgs, 1:length(error_d_res_short_iters["32"]), error_d_res_short_iters[string(3,2)], yaxis=:log10, label = string(strategies_short_name[3]))
plot!(error_lbfgs, 1:length(error_d_res_short_iters["42"]), error_d_res_short_iters[string(4,2)], yaxis=:log10, label = string(strategies_short_name[4]))
plot!(error_lbfgs, 1:length(error_d_res_short_iters["52"]), error_d_res_short_iters[string(5,2)], yaxis=:log10, label = string(strategies_short_name[5]))
plot!(error_lbfgs, 1:length(error_d_res_short_iters["62"]), error_d_res_short_iters[string(6,2)], yaxis=:log10, label = string(strategies_short_name[6]))

Plots.plot(error_adam, error_lbfgs, layout = Plots.grid(1, 2, widths=[0.5 ,0.5]), size = (1500,500))
Plots.savefig("./Results/Plots/Error vs Iters/Diffusion/Diffusion_ADAM_vs_LBFGS_50_iter_reltol_1.pdf")



##Prediction plot

@parameters x t
domains = [x ∈ IntervalDomain(-1.0,1.0),
           t ∈ IntervalDomain(0.0,1.0)]
dx = 0.2; dt = 0.1
xs,ts = [domain.domain.lower:dx/10:domain.domain.upper for (dx,domain) in zip([dx,dt],domains)]
analytic_sol_func(x,t) =  sin(pi*x) * exp(-t)
u_real = reshape([analytic_sol_func(x,t) for x in xs for t in ts], (length(xs),length(ts)))


u_predict = Dict()
diff_u = Dict()
total_errors = Dict()
for strat in 1:6
      for min in 1:1
            chain = FastChain(FastDense(2,18,Flux.σ),FastDense(18,18,Flux.σ),FastDense(18,1))
            discretization = PhysicsInformedNN(chain,strategies[strat])
            phi = discretization.phi
            u_predict[strat,min] = reshape([first(phi([x,t],pars_d_res[string(strat,min)])) for x in xs for t in ts],(length(xs),length(ts)))
            diff_u[strat,min] = abs.(u_predict[strat,min] .- u_real)
            #push!(total_errors , mean(diff_u))
            #push!(u_predicts, u_predict)
            #push!(diff_us, diff_u)
      end
end

diff_u

#names  = ["GridTraining","StochasticTraining", "QuasiRandomTraining", "" ]
#for (u_predict, diff_u,strategy,name) in zip(u_predicts,diff_us,strategies,names)
for strat in 1:6
      for min in 1:1
            p1 = plot(xs, ts, u_real, linetype=:contourf,title = "Analytic",clims = (-1,1));
            p2 = plot(xs, ts, u_predict[strat,min], linetype=:contourf,title = string(strategies_short_name[strat], " / ", minimizers_short_name[min]), clims=(-1,1));
            p3 = plot(xs, ts, diff_u[strat,min],linetype=:contourf,title = "Error");
            plot(p1,p2,p3, layout = Plots.grid(1, 3), size = (1500,300));
            savefig(string("./Results/Predictions_diffusion/predict_",strategies_short_name[strat],"_",minimizers_short_name[min],".pdf"))
      end
end

println(total_errors)
