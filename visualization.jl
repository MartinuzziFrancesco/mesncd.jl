using DelimitedFiles, CairoMakie, Statistics, ChaoticDynamicalSystemLibrary,
    ReservoirComputing, Colors, JSON, Random, mesncd

Random.seed!(42)

function blend_colors(c1::RGB, c2::RGB, t::Float64)
    return RGB(
        (1 - t) * c1.r + t * c2.r,
        (1 - t) * c1.g + t * c2.g,
        (1 - t) * c1.b + t * c2.b
    )
end

results = JSON.parsefile("results/results_by_init.json")

ordered_reservoir_inits = [
    "cycle_jumps",
    "delay_line",
    "double_cycle",
    "selfloop_delayline_backward",
    "forward_connection",
    "selfloop_cycle",
    "selfloop_forward_connection",
    "simple_cycle",
    "selfloop_feedback_cycle",
    "esn",
    "delay_line_backward"
]
reversed_inits = reverse(ordered_reservoir_inits)
init_labels = Dict(
    "cycle_jumps" => "CJ",
    "delay_line" => "DL",
    "double_cycle" => "DC",
    "selfloop_delayline_backward" => "SLDB",
    "forward_connection" => "FC",
    "selfloop_cycle" => "SLC",
    "selfloop_forward_connection" => "SLFC",
    "simple_cycle" => "SC",
    "selfloop_feedback_cycle" => "SLFB",
    "esn" => "ESN",
    "delay_line_backward" => "DLB"
)
reversed_labels = [init_labels[init] for init in reversed_inits]

base_color = color_palette[1]
gray_target = RGB(0.85, 0.85, 0.85)
non_esn_colors = [blend_colors(base_color, gray_target, i / 11) for i in 1:10]
esn_color = color_palette[2]
color_map = Dict{String,RGB}()
non_esn_inits = filter(init -> init != "esn", reversed_inits)
for (i, init) in enumerate(reverse(non_esn_inits))
    color_map[init] = non_esn_colors[i]
end
color_map["esn"] = esn_color

fig = Figure(resolution=(900, 1600))

mainf = fig[1, 1] = GridLayout()
errfig = mainf[1, 1] = GridLayout()
stdfig = mainf[1, 2] = GridLayout()

ax = Axis(
    errfig[1, 1],
    #title="Absolute Error",
    #ylabel="Initializers",
    xlabel="CDE",
    yticks=(1:length(reversed_labels), reversed_labels)
)
rng = MersenneTwister(14)
for (i, init) in enumerate(reversed_inits)
    result = results[init]
    errors = result["errors"]
    med = result["median"]
    lower = result["lower_error"]
    upper = result["upper_error"]
    color = color_map[init]

    xvals = fill(i, length(errors)) .+ 0.55 .* (rand(rng, length(errors)) .- 0.5)
    scatter!(ax, errors, xvals; color=color,
        transparency=true, markersize=20)
    scatter!(ax, [med], [i],
        color=:black, markersize=54)
    errorbars!(ax, [med], [i], [lower], [upper]; direction=:x,
        color=:black, linewidth=11, linecap=:round)
end

ax = Axis(
    stdfig[1, 1],
    #title="Absolute Error",
    #ylabel="Reservoir Init",
    xlabel="SD",
    yticklabelsvisible=false,
    yticks=(1:length(reversed_labels), reversed_labels)
)
#rng = MersenneTwister(14)
for (i, init) in enumerate(reversed_inits)
    result = results[init]
    stds = result["stds"]
    med = result["median_std"]
    lower = result["lower_error_std"]
    upper = result["upper_error_std"]
    color = color_map[init]

    xvals = fill(i, length(stds)) .+ 0.55 .* (rand(rng, length(stds)) .- 0.5)
    scatter!(ax, stds, xvals; color=color,
        transparency=true, markersize=20)
    scatter!(ax, [med], [i],
        color=:black, markersize=56)
    errorbars!(ax, [med], [i], [lower], [upper]; direction=:x,
        color=:black, linewidth=11, linecap=:round)
end

for (label, layout) in zip(["(a)", "(b)"], [errfig, stdfig])
    Label(layout[1, 1, TopLeft()], label,
        fontsize=36,
        font=:bold,
        padding=(10, 10, 10, 10),
        halign=:left)
end

colgap!(mainf, 0)
fig

save("figures/fig01.eps", fig, dpi=600)
save("figures/fig01.png", fig, dpi=600)
