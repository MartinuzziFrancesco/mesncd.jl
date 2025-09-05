using DelimitedFiles, Statistics, ChaoticDynamicalSystemLibrary,
ReservoirComputing, Colors, JSON
include("systems.jl")


train_len = 7000
reservoir_inits = [
    delay_line,
    double_cycle,
    selfloop_delayline_backward,
    forward_connection,
    selfloop_cycle,
    selfloop_forward_connection,
    simple_cycle,
    selfloop_feedback_cycle,
    delay_line_backward
]

results = Dict()

for init in reservoir_inits
    mcds = zeros(length(chaos_systems))  # Mean errors
    stds = zeros(length(chaos_systems))  # Std deviations

    for (idx, csystem) in enumerate(chaos_systems)
        system_name = string(csystem)
        filename = "results/min_esn/esn_results_$(init)$(system_name)$(train_len).csv"
        mmetrs = readdlm(filename, ',')
        mcds[idx] = mean(mmetrs)
        stds[idx] = std(mmetrs)
    end

    # Compute stats for means
    med = median(mcds)
    q1 = quantile(mcds, 0.25)
    q3 = quantile(mcds, 0.75)
    lower = med - q1
    upper = q3 - med

    # Compute stats for stds
    mean_std = mean(stds)
    med_std = median(stds)
    q1_std = quantile(stds, 0.25)
    q3_std = quantile(stds, 0.75)
    lower_std = med_std - q1_std
    upper_std = q3_std - med_std

    results[string(init)] = Dict(
        "errors" => mcds,
        "stds" => stds,
        "median" => med,
        "lower_error" => lower,
        "upper_error" => upper,
        "mean_std" => mean_std,
        "median_std" => med_std,
        "lower_error_std" => lower_std,
        "upper_error_std" => upper_std
    )
end

stds = zeros(length(chaos_systems))
cds = zeros(length(chaos_systems))

for (idx, csystem) in enumerate(chaos_systems)
    system_name = string(csystem)
    metrs = readdlm("results/esn/esn_results$system_name.csv", ',')
    mean_metrs = mean(metrs, dims=2)
    std_metrs = std(metrs, dims=2)
    cds[idx] = mean_metrs[2]
    stds[idx] = std_metrs[2]
end

# Compute statistics for mean errors
med = median(cds)
q1 = quantile(cds, 0.25)
q3 = quantile(cds, 0.75)
lower = med - q1
upper = q3 - med

# Compute statistics for stds
mean_std = mean(stds)
med_std = median(stds)
q1_std = quantile(stds, 0.25)
q3_std = quantile(stds, 0.75)
lower_std = med_std - q1_std
upper_std = q3_std - med_std

# Store in structured dict
results["esn"] = Dict(
    "errors" => cds,
    "stds" => stds,
    "median" => med,
    "lower_error" => lower,
    "upper_error" => upper,
    "mean_std" => mean_std,
    "median_std" => med_std,
    "lower_error_std" => lower_std,
    "upper_error_std" => upper_std
)

cds = zeros(length(chaos_systems))     # Mean errors (2nd row)
stds = zeros(length(chaos_systems))    # Std deviations (2nd row)

for (idx, csystem) in enumerate(chaos_systems)
    system_name = string(csystem)
    metrs = readdlm("results/min_esn/esn_results_cycle_jumps$system_name$train_len.csv", ',')
    mean_metrs = mean(metrs, dims=2)
    std_metrs = std(metrs, dims=2)
    cds[idx] = mean_metrs[2]
    stds[idx] = std_metrs[2]
end

# Compute statistics for mean errors
med = median(cds)
q1 = quantile(cds, 0.25)
q3 = quantile(cds, 0.75)
lower = med - q1
upper = q3 - med

# Compute statistics for stds
mean_std = mean(stds)
med_std = median(stds)
q1_std = quantile(stds, 0.25)
q3_std = quantile(stds, 0.75)
lower_std = med_std - q1_std
upper_std = q3_std - med_std

# Store in structured dict
results["cycle_jumps"] = Dict(
    "errors" => cds,
    "stds" => stds,
    "median" => med,
    "lower_error" => lower,
    "upper_error" => upper,
    "mean_std" => mean_std,
    "median_std" => med_std,
    "lower_error_std" => lower_std,
    "upper_error_std" => upper_std
)


open("results/results_by_init.json", "w") do f
    JSON.print(f, results)
end



##### results for viz_attractors
init = cycle_jumps
train_len = 7000
path = "results/min_esn/params/"
param_groups = Dict{String, Vector{String}}()
for (idx, csystem) in enumerate(chaos_systems)
    system_name = string(csystem)
    file = path*"esn_params_$init$system_name$train_len.json"
    params = JSON.parsefile(file)
    param_key = JSON.json(params)
    if haskey(param_groups, param_key)
        push!(param_groups[param_key], system_name)
    else
        param_groups[param_key] = [system_name]
    end
end

num_systems_plot = 5

plot_param_groups = Dict(
    k => v for (k, v) in param_groups if length(v) ≥ num_systems_plot
)

plot_param_groups["{\"reg\":1.0e-10}"]

##### results fo viz_scalelaws

train_lengths = reverse([6500, 6000, 5500, 5000, 4500, 4000, 3500, 3000, 2500, 2000])

init = cycle_jumps
results_cj_tl = Dict()

for train_len in train_lengths
    cds = zeros(length(chaos_systems))
    mcds = zeros(length(chaos_systems))
    for (idx, csystem) in enumerate(chaos_systems)
        system_name = string(csystem)
        filename = "results/min_esn/esn_results_$init$system_name$train_len.csv"
        mmetrs = readdlm(filename, ',')
        mean_mmetrs = mean(mmetrs)
        mcds[idx] = mean_mmetrs
    end
    med = median(mcds)
    q1 = quantile(mcds, 0.25)
    q3 = quantile(mcds, 0.75)
    lower = med - q1
    upper = q3 - med
    results_cj_tl[string(train_len)] = Dict(
        "errors" => mcds,
        "median" => med,
        "lower_error" => lower,
        "upper_error" => upper
    )
end


results_esn_tl = Dict()

for train_len in train_lengths
    cds = zeros(length(chaos_systems))
    mcds = zeros(length(chaos_systems))
    for (idx, csystem) in enumerate(chaos_systems)
        system_name = string(csystem)
        filename = "results/esn/esn_results_$system_name$train_len.csv"
        mmetrs = readdlm(filename, ',')
        mean_mmetrs = mean(mmetrs)
        mcds[idx] = mean_mmetrs
    end
    med = median(mcds)
    q1 = quantile(mcds, 0.25)
    q3 = quantile(mcds, 0.75)
    lower = med - q1
    upper = q3 - med
    results_esn_tl[string(train_len)] = Dict(
        "errors" => mcds,
        "median" => med,
        "lower_error" => lower,
        "upper_error" => upper
    )
end



# Threshold for "successful" reconstructions
threshold = 0.5

cj_errors = Float64.(results["cycle_jumps"]["errors"])
rand_errors = Float64.(results["esn"]["errors"])

cj_success_rate = count(<(threshold), cj_errors) / length(cj_errors)
rand_success_rate = count(<(threshold), rand_errors) / length(rand_errors)

println("CJ success rate: ", cj_success_rate)
println("Rand success rate: ", rand_success_rate)



###### for viz_params

train_len = 7000
path = "results/esn/params/"

param_groups_esn = Dict{String, Vector{String}}()

for csystem in chaos_systems
    system_name = string(csystem)
    file = path * "esn_params$system_name.json"
    params = JSON.parsefile(file)

    hyperparam_key = Dict(
        "reg" => params["reg"],
        "radius" => params["radius"],
        "sparsity" => params["sparsity"]
    )

    key_str = JSON.json(hyperparam_key)

    if haskey(param_groups_esn, key_str)
        push!(param_groups_esn[key_str], system_name)
    else
        param_groups_esn[key_str] = [system_name]
    end
end


topologies = [
    "cycle_jumps", "delay_line", "double_cycle",
    "selfloop_delayline_backward", "forward_connection",
    "selfloop_cycle", "selfloop_forward_connection",
    "simple_cycle", "selfloop_feedback_cycle", "delay_line_backward"
]

train_len = 7000
base_path = "results/min_esn/params/"
reuse_distributions = Dict{String, Dict{Int, Int}}()

for init in topologies
    param_groups = Dict{String, Vector{String}}()
    
    for csystem in chaos_systems
        system_name = string(csystem)
        file = base_path * "esn_params_$init$system_name$train_len.json"
        params = JSON.parsefile(file)
        key_str = JSON.json(params)  # should be just {"reg": val}
        
        if haskey(param_groups, key_str)
            push!(param_groups[key_str], system_name)
        else
            param_groups[key_str] = [system_name]
        end
    end
    
    # Count reuse frequencies
    reuse_counts = countmap(length.(values(param_groups)))
    reuse_distributions[init] = reuse_counts
end