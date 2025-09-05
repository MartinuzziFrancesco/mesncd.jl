function run_best_esn(best_params, input_data, output_data, predict_len;
    n_runs=100, prediction_type=Generative)
    outputs = zeros(Float32, predict_len, n_runs)
    esn = train_esn(input_data, output_data; best_params...)
    return esn[1](prediction_type(predict_len), esn[2])
end

function run_best_esn_normal(rng, best_params, input_data, output_data, predict_len;
    n_runs=100, prediction_type=Generative)
    outputs = zeros(Float32, predict_len, n_runs)
    esn = train_esn_normal(rng, input_data, output_data; best_params...)
    return esn[1](prediction_type(predict_len), esn[2])
end

#@everywhere
function grid_search_esn(input_data, output_data, param_grid, n_folds::Int=1;
    metric=smape, val_size=350, prediction_type=Generative, verbose=true, kwargs...)
    best_params = nothing
    best_performance = Inf

    if verbose
        println("")
        println("==> Running temporal cross validation with ", n_folds, " folds <==")
        println("")
    end

    param_combinations = collect(Iterators.product(values(param_grid)...))
    Threads.@threads for params in ProgressBar(param_combinations)
        fold_params = Dict{Symbol,Any}(zip(keys(param_grid), params))
        if n_folds > 1
            performance = temporal_cross_validation(
                input_data,
                output_data,
                fold_params,
                n_folds,
                val_size;
                prediction_type=prediction_type,
                kwargs...,
            )
        else
            esn = train_esn(input_data, output_data; fold_params...)
            performance = evaluate_esn(
                esn, output_data; metric=metric, prediction_type=prediction_type
            )
        end

        if performance < best_performance
            best_performance = performance
            best_params = fold_params
        end
    end

    return best_params
end

function run_parallel_grid_search(input_data_list, output_data_list, param_grid, n_folds::Int=1; kwargs...)
    results = pmap((x, y) -> grid_search_esn(x, y, param_grid, n_folds; kwargs...), zip(input_data_list, output_data_list))
    return results
end



function temporal_cross_validation(input_data, output_data, params, n_folds,
    val_size=350; kwargs...)
    performances = zeros(n_folds)
    for fold in 1:n_folds
        train_index_end = div(fold * size(input_data, 2), n_folds) - val_size
        val_index_start = train_index_end + 1
        val_index_end = min(div(fold * size(input_data, 2), n_folds), train_index_end + val_size)  # Ensure it does not exceed fold end
        train_input = input_data[:, 1:train_index_end]
        train_output = output_data[:, 1:train_index_end]
        val_output = output_data[:, val_index_start:val_index_end]
        esn = train_esn(train_input, train_output; params...)
        performance = evaluate_esn(esn, val_output; kwargs...)
        performances[fold] = performance
    end
    avg_performance = mean(performances)
    return avg_performance
end

function evaluate_esn(esn, target_output; metric=smape, prediction_type=Generative)
    predict_len = size(target_output, 2)
    output = esn[1](prediction_type(predict_len), esn[2])
    acc = metric(output, target_output)
    return acc
end

#=
function train_esn(input_data::AbstractArray, output_data::AbstractArray;
        res_size::Int=300, radius::Number=1.0, sparsity::Number=0.1,
        reg::Number=0.0, leaky_coefficient::Number = 1.0, scaling=0.1)
    esn = ESN(input_data, size(input_data, 1), res_size;
        reservoir_driver = RNN(; leaky_coefficient = leaky_coefficient),
        reservoir=rand_sparse(; radius = radius, sparsity = sparsity),
        input_layer = weighted_init(; scaling = scaling),
        nla_type=NLAT2()
    )
    output_layer = ReservoirComputing.train(esn, output_data, StandardRidge(reg))
    return esn, output_layer
end
=#


function train_esn(input_data::AbstractArray, output_data::AbstractArray;
    res_size::Int=300, init_reservoir=cycle_jumps,
    reg::Number=0.0, leaky_coefficient::Number=1.0)
    esn = ESN(input_data, size(input_data, 1), res_size;
        reservoir_driver=RNN(; leaky_coefficient=leaky_coefficient),
        reservoir=init_reservoir,
        input_layer=minimal_init(; weight=0.01, sampling_type=:irrational_sample!),
        states_type=PaddedStates(),
        nla_type=NLAT2(),
        bias=ones32
    )
    output_layer = ReservoirComputing.train(esn, output_data, StandardRidge(reg))
    return esn, output_layer
end


function train_esn_normal(rng, input_data::AbstractArray, output_data::AbstractArray;
    res_size::Int=300, radius::Number=1.0, sparsity::Number=0.1,
    reg::Number=0.0, leaky_coefficient::Number=1.0, scaling=0.1)
    esn = ESN(input_data, size(input_data, 1), res_size;
        reservoir_driver=RNN(; leaky_coefficient=leaky_coefficient),
        reservoir=rand_sparse(; radius=radius, sparsity=sparsity),
        input_layer=weighted_init(; scaling=scaling),
        nla_type=NLAT2(),
        rng=rng
    )
    output_layer = ReservoirComputing.train(esn, output_data, StandardRidge(reg))
    return esn, output_layer
end
