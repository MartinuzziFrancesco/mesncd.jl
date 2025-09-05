using ReservoirComputing, OrdinaryDiffEq, Plots, Random, SparseArrays,
StatsBase, ChaoticDynamicalSystemLibrary, LinearAlgebra, FractalDimensions,
DelimitedFiles, ProgressBars, JSON, CairoMakie

CairoMakie.activate!(type = "png")

include("systems.jl")
include("data.jl")
include("training.jl")
include("evaluation.jl")


rng = Random.default_rng()
Random.seed!(rng, 17)

shift = 300
train_len = 7000
predict_len = 1250
n_ic = 20

prob = ChaoticDynamicalSystemLibrary.CaTwoPlus()
system_name = "CaTwoPlus"
dt, data_tr, data = data_from_ode(rng, prob)
input_data, target_data, test = split_data(data, shift, train_len, predict_len)
esn_params = grid_search_esn(input_data, target_data, esn_param_grid, 4)

output = run_best_esn(esn_params, input_data, target_data, predict_len)
test = StatsBase.reconstruct(data_tr, test)
output = StatsBase.reconstruct(data_tr, output)
err = incremental_smape(test, output)
vpt = valid_prediction_time(system_name, dt, err)
cd = compare_corr_dim(test, output)


fig = Figure(size=(3508, 2480),
            fontsize=28,
            backgroundcolor = RGBf(0.145, 0.145, 0.145))

ax = Axis(fig[1, 1])

lines!(ax, collect(1:predict_len), test[1,:], 
    linewidth=4.0,
    color=:black)
lines!(ax, collect(1:predict_len), output[1,:], 
    linewidth=4.0,
    color=:red)

