using ReservoirComputing, OrdinaryDiffEq, Plots, Random, SparseArrays,
    StatsBase, ChaoticDynamicalSystemLibrary, LinearAlgebra, FractalDimensions,
    DelimitedFiles, ProgressBars, JSON, mesncd

rng = Random.default_rng()
Random.seed!(rng, 17)

shift = 300
train_len = 7000
predict_len = 2500
n_ic = 20

esn_param_grid = Dict(
    :radius => [0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
    #:reg => [1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    :reg => [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    :sparsity => [0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
)

for csystem in chaos_systems
    #for init in reservoir_inits
    global shift, predict_len, n_ic, esn_param_grid, rng, train_len
    shift = 300
    system_name = string(csystem)
    largest_lyap = ChaoticDynamicalSystemLibrary.ATTRACTOR_DATA[system_name]["maximum_lyapunov_estimated"]
    println("Currently tackling: ", system_name)
    prob = csystem()
    dt, data_tr, data = data_from_ode(rng, prob)
    input_data, target_data, test = split_data(data, shift, train_len, predict_len)
    esn_params = grid_search_esn(input_data, target_data, esn_param_grid, 4)
    #esn_params[:init_reservoir] = init

    esn_results = zeros(n_ic)
    for irng in 1:n_ic
        shift += 50
        input_data, target_data, test = split_data(data, shift, train_len, predict_len)
        output = run_best_esn(esn_params, input_data, target_data, predict_len)
        test = StatsBase.reconstruct(data_tr, test)
        output = StatsBase.reconstruct(data_tr, output)
        cd = compare_corr_dim(test, output)
        esn_results[irng] = cd
    end
    #pop!(esn_params, :init_reservoir)
    open("results/esn/esn_results_$system_name$train_len.csv", "w") do io
        writedlm(io, esn_results, ',')
    end

    open("results/esn/params/esn_params_$system_name$train_len.json", "w") do io
        JSON.print(io, esn_params)
    end
end
