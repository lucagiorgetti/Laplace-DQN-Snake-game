#structs.jl
######################################################################################
#Improves perforance using two frames as a single state
#######################environment object############################################

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

    function DQNModel(game::SnakeGame, model_rng = Xoshiro(42))
        board_size = game.board_size
        n_actions = 3

	q_net = Chain(
    		# now 2 ⇒ 16 channels
    		Conv((3, 3), 2 => 16, relu; pad=(1,1)),                         #304 params
    		Conv((3, 3), 16 => 32, relu; pad=(1,1)),                        #4 640 params
    		Conv((6, 6), 32 => 64, relu),                                   #73 792 params
    		Flux.flatten,                                                  
    		Dense((board_size - 5)*(board_size - 5)*64, 64, relu),          #102 464 params
    		Dense(64, n_actions)                                            #195 params
		)

        t_net = deepcopy(q_net)
        opt = RMSProp(0.0005)   

        new(q_net, t_net, opt)
    end
end

########################################wrapper object########################################################
#no modifications
mutable struct Trainer
    game::SnakeGame
    model::DQNModel
    buffer::ReplayBuffer
    n_batches::Int
    target_update_rate::Int
    epsilon::Float32
    epsilon_end::Float32
    decay::Float32
    save::Bool
    losses::Vector{Float32}
    episode_rewards::Vector{Float32}

    function Trainer(; n_batches::Int = 1000, target_update_rate::Int = 1000,
                     epsilon::Float32 = 1.0f0, epsilon_end::Float32 = 0.05f0, decay::Float32= 0.00001f0, save::Bool = true,
                     game::SnakeGame = SnakeGame(),
                     model::Union{DQNModel, Nothing} = nothing)
                     
        model = isnothing(model) ? DQNModel(game) : model
        buffer = ReplayBuffer()
        losses = Float32[]
        episode_rewards = Float32[]
        new(game, model, buffer, n_batches, target_update_rate, epsilon, epsilon_end, decay, save, losses, episode_rewards)
    end
end

