using Images, ColorTypes, Plots
using Random
using StatsBase
using Flux
using Flux.Optimise: RMSProp
using Printf

#type alias
const Experience = Tuple{Matrix{Int64}, CartesianIndex{2}, Float32, Matrix{Int64}, Bool}

const ACTIONS = Dict(
    CartesianIndex(-1, 0) => 1,  # Up
    CartesianIndex(1, 0)  => 2,  # Down
    CartesianIndex(0, -1) => 3,  # Left
    CartesianIndex(0, 1)  => 4   # Right
)

include("env.jl")      # defines SnakeGame
include("model.jl")    # defines DQNModel (needs SnakeGame)
include("buffer.jl")   # defines ReplayBuffer (needs DQNModel for fill_buffer!)
include("train.jl")    # defines train! (needs everything)

