#imports.jl
################################################################################################
#new imports.jl script, now experience is defined as Array{Matrix{Int}]: is a stacking of 2 frames
#################################################################################################
using Images, ColorTypes, Plots
using Random
using StatsBase
using Flux
using Flux.Optimise: RMSProp
using Printf
using BSON: @save, @load
using Logging
using TerminalLoggers
using Optimisers
using Distributions
using LinearAlgebra
using Mmap
using GameZero
#using FileIO, JLD2

#debug environment
ENV["JULIA_DEBUG"] = Main


#type alias
const Experience = Tuple{
    Array{Int,4},                # 1.state
    CartesianIndex{2},           # 2.action taken
    Float64,                     # 3.reward
    Array{Int,4},                # 4.next state
    Bool,                        # 5.done
    Vector{CartesianIndex{2}},   # 6.available actions at time of action
    Vector{CartesianIndex{2}},   # 7.available actions at the next state
    Vector{Bool}                 # 8.if available actions are suicidal                      
}

include("structs.jl")     
include("utils.jl")   
#include("la_utils.jl")
