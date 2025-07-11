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

#type alias
const Experience = Tuple{
    Array{Int,4},                # state
    CartesianIndex{2},           # action taken
    Float64,                     # reward
    Array{Int,4},                # next state
    Bool,                        # done
    Vector{CartesianIndex{2}},   # available actions at time of action
    Vector{CartesianIndex{2}},   # available actions at the next state
    Vector{Bool}                 # if available actions are suicidal                      
}

mutable struct SnakeGame
    board_size::Int
    n_frames::Int
    board::Matrix{Int}
    board_history::Vector{Matrix{Int}}
    action_history::Vector{CartesianIndex{2}}
    reward_history::Vector{Float32}
    done_history::Vector{Bool}
    av_action_history::Vector{Vector{CartesianIndex{2}}}
    av_next_action_history::Vector{Vector{CartesianIndex{2}}}
    next_is_suicidal_history::Vector{Vector{Bool}}
    state::Array{Int,4}
    snake::Vector{CartesianIndex{2}} #head is the first tuple, queue the last
    direction::CartesianIndex{2}     #It will have to be initialized using DQN
    prev_dir::CartesianIndex{2}
    score::Int
    reward::Float32
    #defining some constants for clarity
    eating_reward::Float32
    suicide_penalty::Float32
    male_di_vivere::Float32
    food_rng::AbstractRNG
    lost::Bool
    discount::Float32
    food_list::Vector{CartesianIndex{2}}  

    # Custom constructor
    function SnakeGame(board_size = 10, n_frames = 2, discount = 0.99, food_rng = Xoshiro(42))
        board = zeros(Int, (board_size, board_size))  # Create empty grid

        # Draw walls (-1)
        board[1, :] .= -1
        board[end, :] .= -1
        board[:, 1] .= -1
        board[:, end] .= -1
        
        #initialize food
        board[4, 5] = 2

        # Initialize the snake (bottom-left)
        
        snake = [CartesianIndex(board_size - 2, 2),CartesianIndex(board_size - 1, 2)]
        
        for ci in snake 
        	board[ci] = 1
        end 
        
        board_history = [deepcopy(board) for _ in 1:n_frames]
        state = cat(board_history...;dims = 3)
        state = reshape(state, board_size, board_size, n_frames, 1)
        
        action_history = CartesianIndex{2}[]
        reward_history = Float32[]
        done_history = Bool[]
        av_action_history = CartesianIndex{2}[]
        av_next_action_history = Float32[]
        next_is_suicidal_history = Array{Bool,4}[]
        
        #initialize direction (for now)
        direction = CartesianIndex(0,0)   # this will be overwritten 
        prev_dir = CartesianIndex(-1,0)   #move up, this is important: I need to check if the next move is not move down, otherwise game is lost
        
        lost = false
        
        food_list = [CartesianIndex(rand(food_rng, 2: board_size - 1), rand(food_rng, 2: board_size - 1)) for _ in 1:50]
        
        new(
    	    board_size,
            n_frames,
            board,
            board_history,
            action_history,
            reward_history,
            done_history,
            av_action_history,
            av_next_action_history,          
            next_is_suicidal_history,        
            state,                           
            snake,
            direction,
            prev_dir,
            0,                               # score
            0.0f0,                           # reward
            1.0f0,                           # eating_reward
            -1.0f0,                          # suicide_penalty
            -0.01f0,                         # male_di_vivere
            food_rng,
            lost,
            discount,
            food_list
            )


    end
end

gm = SnakeGame()

WIDTH = 600
HEIGHT = 600
BACKGROUND = colorant"antiquewhite"

#unit, dimension of a pixel
u = 60

#define the walls actors
walls = [
    Rect(0, 0, WIDTH, u),              # top wall
    Rect(0, HEIGHT - u, WIDTH, u),     # bottom wall
    Rect(0, 0, u, HEIGHT),             # left wall
    Rect(WIDTH - u, 0, u, HEIGHT)      # right wall
]
wall_color = colorant"black"

# Snake head starts at bottom-left
snake_head = Rect(u, HEIGHT - 3*u, u, u)

# Body: one additional segment below the head
snake_body = [
    Rect(u, HEIGHT - 3*u, u, u),   # head (redundant but helps generalize)
    Rect(u, HEIGHT - 2*u, u, u)     # tail segment directly below
]

snake_color = colorant"green"

#define apple actor
apple = Rect(4*u,3*u,u,u)

apple_color = colorant"red"



function draw(g::Game)
    # background and walls
    for wall in walls
        draw(wall, wall_color, fill=true)
    end

    # snake
    draw(snake_head, snake_color, fill=true)
    for part in snake_body
        draw(part, snake_color, fill=true)
    end
    
    draw(apple, apple_color, fill=true)
end



