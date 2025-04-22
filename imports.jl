using Images, ColorTypes, Plots
using Random
using StatsBase
using Flux
using Flux.Optimise: RMSProp
using Printf
using BSON: @save, @load

#type alias
const Experience = Tuple{Matrix{Int64}, CartesianIndex{2}, Float32, Matrix{Int64}, Bool}

const ACTIONS = Dict(
    CartesianIndex(-1, 0) => 1,  # Up
    CartesianIndex(1, 0)  => 2,  # Down
    CartesianIndex(0, -1) => 3,  # Left
    CartesianIndex(0, 1)  => 4   # Right
)

include("structs.jl")     
include("utils.jl")   
