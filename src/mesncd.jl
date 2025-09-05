module mesncd

using ReservoirComputing
using OrdinaryDiffEq
using Random
using StatsBase
using ChaoticDynamicalSystemLibrary
using LinearAlgebra
using FractalDimensions
using DelimitedFiles
using ProgressBars
using JSON
using Statistics
using Colors
using CairoMakie


include("data.jl")
include("evaluation.jl")
include("systems.jl")
include("theme.jl")
include("training.jl")

export data_from_ode, data_from_ode_standard, split_data
export smape, rmse_scalar, compare_corr_dim
export chaos_systems
export fullpage_theme
export run_best_esn, run_best_esn_normal, grid_search_esn

end
