using DelimitedFiles, CairoMakie, Statistics, ChaoticDynamicalSystemLibrary,
    ReservoirComputing, Colors, JSON, Random, OrdinaryDiffEq, StatsBase, LaTeXStrings,
    FractalDimensions, mesncd

Random.seed!(42)


function blend_colors(c1::RGB, c2::RGB, t::Float64)
    return RGB(
        (1 - t) * c1.r + t * c2.r,
        (1 - t) * c1.g + t * c2.g,
        (1 - t) * c1.b + t * c2.b
    )
end

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
    "esn" => "Rand",
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

chosen_system = ChaoticDynamicalSystemLibrary.Arneodo
largest_lyap = ChaoticDynamicalSystemLibrary.ATTRACTOR_DATA["Arneodo"]["maximum_lyapunov_estimated"]

prob = chosen_system()
dt, data_tr, data = data_from_ode_standard(prob)


param_range = reverse([1.0e-10, 1.0e-11, 1.0e-12, 1.0e-13, 1.0e-14])

shift = 300
train_len = 7000
predict_len = 900#2500
add_names = true

input_data, target_data, test = split_data(
    data, shift, train_len, predict_len
)

fig = Figure(resolution=(1100, 1300))

mainf = fig[1, 1] = GridLayout()
mesn1 = mainf[1, 1] = GridLayout()
mesn2 = mainf[2, 1] = GridLayout()
mesn3 = mainf[3, 1] = GridLayout()
mesn4 = mainf[4, 1] = GridLayout()
mesn5 = mainf[5, 1] = GridLayout()
focusf = mainf[6, 1] = GridLayout()
plot_axis = [mesn1, mesn2, mesn3, mesn4, mesn5]

# Reduce gaps inside each subplot
#GridLayout(; colgap=30, rowgap=30)

for (idx, reg_param) in enumerate(param_range)
    esn_params = Dict(
        :reg => reg_param,
        :init_reservoir => cycle_jumps
    )
    output = run_best_esn(
        esn_params, input_data, target_data, predict_len
    )
    dd = compare_corr_dim(test, output)
    println(dd)
    #for var in 1:3
    var = 2
    if idx != 5
        ax = Axis(plot_axis[idx][1, 2],
            rightspinevisible=false,
            leftspinevisible=false,
            topspinevisible=false,
            bottomspinevisible=false,
            yticksvisible=false,
            xticksvisible=false,
            xticklabelsvisible=false,
            yticklabelsvisible=false
        )
    else
        ax = Axis(plot_axis[idx][1, 2],
            rightspinevisible=false,
            leftspinevisible=false,
            topspinevisible=false,
            yticksvisible=false,
            xticksvisible=false,
            bottomspinevisible=false,
            #xticklabelsvisible=false,
            yticklabelsvisible=false,
            xlabel=L"\lambda_{max} t",
            xticklabelsize=36,
            xlabelsize=45,
        )
    end

    pred_len = length(output[var, :])
    normal_t = dt .* (0:pred_len-1)
    lyap_t = largest_lyap .* normal_t

    lines!(ax, lyap_t, test[var, :],
        color=RGB(0.55, 0.55, 0.55), linewidth=6)
    lines!(ax, lyap_t, output[var, :],
        color=color_map["delay_line"], linewidth=6)

    if idx == 5
        x = [0.0, 4.9, 4.9, 0.0, 0.0]
        y = [-50.0, -50.0, 50.0, 50.0, -50.0]
        lines!(ax, x, y, linewidth=4.0, color=color_palette[7])
        ax = Axis(focusf[1, 2],
            leftspinecolor=color_palette[7],
            rightspinecolor=color_palette[7],
            bottomspinecolor=color_palette[7],
            topspinecolor=color_palette[7],
            ytickcolor=color_palette[7],
            xtickcolor=color_palette[7],
            yticklabelsvisible=false,
            xlabel=L"\lambda_{max} t",
            xticklabelsize=36,
            xlabelsize=45,
        )
        lines!(ax, lyap_t[1:400], test[var, 1:400],
            color=RGB(0.55, 0.55, 0.55), linewidth=6)
        lines!(ax, lyap_t[1:400], output[var, 1:400],
            color=color_map["delay_line"], linewidth=6)
        Box(focusf[1, 1], color=:white, strokewidth=0,)
        Label(focusf[1, 1], "focus",
            rotation=pi / 2,
            tellheight=false,
            font=:bold,
            fontsize=38,
            color=:white)
    end
    if idx != 3
        color_label = :black
    else
        color_label = color_palette[3]
    end
    Box(plot_axis[idx][1, 1], color=:white, strokewidth=0,)
    Label(plot_axis[idx][1, 1], string(param_range[idx]),
        rotation=pi / 2,
        tellheight=false,
        font=:bold,
        fontsize=38,
        color=color_label)
    #end
end

for (label, layout) in zip(["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"],
    [mesn1, mesn2, mesn3, mesn4, mesn5, focusf])
    Label(layout[1, 1, TopLeft()], label,
        fontsize=36,
        font=:bold,
        padding=(10, 10, 10, 10),
        halign=:left)
end

fig
#rowgap!(mainf, -150)
rowgap!(mainf, 1, -10)
rowgap!(mainf, 2, -10)
rowgap!(mainf, 3, -10)
rowgap!(mainf, 4, -10)
fig

save("test.png", fig, dpi=600)
save("figures/fig03.eps", fig, dpi=600)
