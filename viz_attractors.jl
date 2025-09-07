using DelimitedFiles, CairoMakie, Statistics, ChaoticDynamicalSystemLibrary,
    ReservoirComputing, Colors, JSON, Random, OrdinaryDiffEq, StatsBase,
    FractalDimensions, mesncd

Random.seed!(42)

base_color = color_palette[2]

soft_gray = RGB(0.95, 0.95, 0.95)

esns_color = RGB(
    0.85 * base_color.r + 0.15 * soft_gray.r,
    0.85 * base_color.g + 0.15 * soft_gray.g,
    0.85 * base_color.b + 0.15 * soft_gray.b
)

esno_color = base_color

five_systems = [
    ChaoticDynamicalSystemLibrary.GenesioTesi,
    ChaoticDynamicalSystemLibrary.Chua,
    ChaoticDynamicalSystemLibrary.SprottS,
    ChaoticDynamicalSystemLibrary.SprottM,
    ChaoticDynamicalSystemLibrary.Rucklidge
]

esn_params = Dict(
    :reg => 1.0e-10,
    :init_reservoir => cycle_jumps
)

shift = 300
train_len = 7000
predict_len = 2000#2500
add_names = true

fig = Figure(resolution=(1900, 1300))

mainf = fig[1, 1] = GridLayout()
minesn = mainf[1, 1] = GridLayout()
esno = mainf[2, 1] = GridLayout()
esns = mainf[3, 1] = GridLayout()
#legend = fig[2,1] = GridLayout()

for (idx, csystem) in enumerate(five_systems)
    system_name = string(csystem)
    println("Currently tackling: ", system_name)
    prob = csystem()
    dt, data_tr, data = data_from_ode_standard(prob)
    input_data, target_data, test = split_data(
        data, shift, train_len, predict_len
    )
    output = run_best_esn(
        esn_params, input_data, target_data, predict_len
    )
    dd = compare_corr_dim(test, output)
    println(dd)
    if add_names
        Box(minesn[1, 1], color=:white, strokewidth=0,)
        Label(minesn[1, 1], "CJ-same",
            rotation=pi / 2,
            tellheight=false,
            font=:bold,
            fontsize=44,)
        #color=RGB(0.4, 0.4, 0.4))
    else
        idx -= 1
    end
    ax = LScene(minesn[1, idx+1], scenekw=(show_axis=false,))

    #ax = Axis3(minesn[1, idx])
    lines!(ax, test[1, :], test[2, :], test[3, :];
        color=RGB(0.85, 0.85, 0.85), linewidth=4)
    lines!(ax, output[1, :], output[2, :], output[3, :];
        color=color_palette[1], linewidth=4)
    cam3d!(ax;
        lookat=Vec3f(0, 0, 0), eyeposition=Vec3f(1, 1, 0.6))
    axis = ax.scene.plots[1]  # the internal 3D axis object
    axis[:showaxis][] = false
    axis[:showticks][] = false
    axis[:showgrid][] = false

    #hidespines!(ax)
    #hidedecorations!(ax)
end

fig

rngs = Dict(
    "GenesioTesi" => 8,#3#8 halfway
    "SprottS" => 1,
    "Rucklidge" => 2,
    "Chua" => 5,#7
    "SprottM" => 1
)

for (idx, csystem) in enumerate(five_systems)
    system_name = string(csystem)
    rng = MersenneTwister(rngs[system_name])
    println("Currently tackling: ", system_name)
    prob = csystem()
    dt, data_tr, data = data_from_ode_standard(prob)
    input_data, target_data, test = split_data(
        data, shift, train_len, predict_len
    )
    esno_params = JSON.parsefile(
        "results/esn/params/esn_params$system_name.json"
    )
    esno_params = Dict(Symbol(k) => v for (k, v) in esno_params)
    output = run_best_esn_normal(
        rng, esno_params, input_data, target_data, predict_len
    )
    dd = compare_corr_dim(test, output)
    println(dd)
    if add_names
        Box(esno[1, 1], color=:white, strokewidth=0,)
        Label(esno[1, 1], "ESN-optimal",
            rotation=pi / 2,
            tellheight=false,
            font=:bold,
            fontsize=44,)
    else
        idx -= 1
    end
    ax = LScene(esno[1, idx+1], scenekw=(show_axis=false,))
    #ax = Axis3(minesn[1, idx])
    lines!(ax, test[1, :], test[2, :], test[3, :];
        color=RGB(0.85, 0.85, 0.85), linewidth=4)
    lines!(ax, output[1, :], output[2, :], output[3, :];
        color=esno_color, linewidth=4)
    cam3d!(ax;
        lookat=Vec3f(0, 0, 0), eyeposition=Vec3f(1, 1, 0.6))
    axis = ax.scene.plots[1]  # the internal 3D axis object
    axis[:showaxis][] = false
    axis[:showticks][] = false
    axis[:showgrid][] = false

    #hidespines!(ax)
    #hidedecorations!(ax)
end

for (idx, csystem) in enumerate(five_systems)
    system_name = string(csystem)
    rng = MersenneTwister(19)
    println("Currently tackling: ", system_name)
    prob = csystem()
    dt, data_tr, data = data_from_ode_standard(prob)
    input_data, target_data, test = split_data(
        data, shift, train_len, predict_len
    )
    esno_params = JSON.parsefile(
        "results/esn/params/esn_paramsSprottS.json"
    )
    esno_params = Dict(Symbol(k) => v for (k, v) in esno_params)
    output = run_best_esn_normal(
        rng, esno_params, input_data, target_data, predict_len
    )
    dd = compare_corr_dim(test, output)
    println(dd)
    if add_names
        Box(esns[1, 1], color=:white, strokewidth=0,)
        Label(esns[1, 1], "ESN-same",
            rotation=pi / 2,
            tellheight=false,
            font=:bold,
            fontsize=44,)
    else
        idx -= 1
    end
    ax = LScene(esns[1, idx+1], scenekw=(show_axis=false,))
    #ax = Axis3(minesn[1, idx])
    lines!(ax, test[1, :], test[2, :], test[3, :];
        color=RGB(0.85, 0.85, 0.85), linewidth=4)
    lines!(ax, output[1, :], output[2, :], output[3, :];
        color=esns_color, linewidth=4)
    cam3d!(ax;
        lookat=Vec3f(0, 0, 0), eyeposition=Vec3f(1, 1, 0.6))
    axis = ax.scene.plots[1]  # the internal 3D axis object
    axis[:showaxis][] = false
    axis[:showticks][] = false
    axis[:showgrid][] = false

    #hidespines!(ax)
    #hidedecorations!(ax)
end

fig

text!(fig.scene, "(a)";
    position=Point2f(100, 1200), fontsize=36, font=:bold)
text!(fig.scene, "(b)";
    position=Point2f(100, 780), fontsize=36, font=:bold)
fig
text!(fig.scene, "(c)";
    position=Point2f(100, 350), fontsize=36, font=:bold)
fig
#for (label, layout) in zip(["(a)", "(b)", "(c)"], [minesn, esno, esns])
#    Label(layout[1, 1, TopLeft()], label,
#        fontsize = 36,
#        font = :bold,
#        padding = (10,10,10,10),
#        halign = :left)
#end


rowgap!(mainf, 1, Fixed(0))
rowgap!(mainf, 2, Fixed(0))
colgap!(minesn, 0)
colgap!(esno, 0)
colgap!(esns, 0)

rowgap!(minesn, 0)
rowgap!(esno, 0)
rowgap!(esns, 0)
fig

save("figures/fig02.eps", fig, dpi=600)
save("figures/fig02.png", fig, dpi=600)
