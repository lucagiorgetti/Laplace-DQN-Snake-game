using Images, ColorTypes, Plots
using Random
using StatsBase
using Flux
using Flux.Optimise: RMSProp
using Printf
using BSON: @save, @load

#type alias
const Experience = Tuple{
    Matrix{Int64},               # state
    CartesianIndex{2},           # action taken
    Float64,                     # reward
    Matrix{Int64},               # next state
    Bool,                        # done
    Vector{CartesianIndex{2}}    # available actions at time of action
}


"""
const ACTIONS = Dict(
    CartesianIndex(-1, 0) => 1,  # Up
    CartesianIndex(1, 0)  => 2,  # Down
    CartesianIndex(0, -1) => 3,  # Left
    CartesianIndex(0, 1)  => 4   # Right
)
"""

include("structs.jl")     
include("utils.jl")   
