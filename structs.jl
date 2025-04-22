#######################environment object############################################àà

mutable struct SnakeGame
    state::Matrix{Int}
    snake::Vector{CartesianIndex{2}} #head is the first tuple, queue the last
    direction::CartesianIndex{2}     #It will have to be initialized using DQN
    prev_move::CartesianIndex{2}
    score::Int
    reward::Float32
    rng::AbstractRNG
    state_size::Int
    lost::Bool
    discount::Float32  

    # Custom constructor
    function SnakeGame(state_size = 10, discount = 0.99, rng = Xoshiro(42))
        state = zeros(Int, (state_size, state_size))  # Create empty grid

        # Draw walls (-1)
        state[1, :] .= -1
        state[end, :] .= -1
        state[:, 1] .= -1
        state[:, end] .= -1

        # Initialize the snake (bottom-left)
        
        snake = [CartesianIndex(state_size - 2, 2),CartesianIndex(state_size - 1, 2)]
        
        for ci in snake 
        	state[ci] = 1
        end 
        
        #initialize direction (for now)
        direction = CartesianIndex(-1,0)  #move up
        prev_move = CartesianIndex(0,0)   #stay still
        
        lost = false
        
        new(state, snake, direction, prev_move, 0, 0, rng, state_size, lost, discount)
    end
end

###################################buffer object##########################################

mutable struct ReplayBuffer
        capacity::Int
        position::Int
        buffer::Vector{Experience}
        batch_size::Int
        
        function ReplayBuffer(capacity = 10000)
                  buffer = Vector{Experience}(undef, 0)
                  batch_size = 64
                  if batch_size > capacity throw("batch_size cannot be greater than the capacity of the buffer.") end
                  new(capacity, 1, buffer, batch_size)
        end
end

##################################model object#########################################################

mutable struct DQNModel
    q_net::Chain
    t_net::Chain
    opt::RMSProp

    function DQNModel(game::SnakeGame)
        board_size = game.state_size

        q_net = Chain(
            Conv((3, 3), 1 => 16, relu; pad=(1,1)), 
            Conv((3, 3), 16 => 32, relu; pad=(1,1)),
            Conv((6, 6), 32 => 64, relu),
            Flux.flatten,
            Dense(((board_size - 5) * (board_size - 5) * 64), 64, relu),
            Dense(64, 4)                 #output n_actions == 4
        )

        t_net = deepcopy(q_net)
        opt = RMSProp(0.0005)   

        new(q_net, t_net, opt)
    end
end

########################################wrapper object########################################################

mutable struct Trainer
    game::SnakeGame
    model::DQNModel
    buffer::ReplayBuffer
    n_batches::Int
    target_update_rate::Int
    epsilon::Float32
    save::Bool
    losses::Vector{Float32}

    function Trainer(; n_batches::Int = 1000, target_update_rate::Int = 100,
                     epsilon::Float32 = 0.8f0, save::Bool = true,
                     game::SnakeGame = SnakeGame(),
                     model::Union{DQNModel, Nothing} = nothing)
                     
        model = isnothing(model) ? DQNModel(game) : model
        buffer = ReplayBuffer()
        losses = Float32[]
        new(game, model, buffer, n_batches, target_update_rate, epsilon, save, losses)
    end
end

              


