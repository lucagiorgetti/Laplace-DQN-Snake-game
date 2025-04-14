#train loop
include("env.jl")
include("buffer.jl")
include("model.jl")

const ACTIONS = Dict(
    CartesianIndex(-1, 0) => 1,  # Up
    CartesianIndex(1, 0)  => 2,  # Down
    CartesianIndex(0, -1) => 3,  # Left
    CartesianIndex(0, 1)  => 4   # Right
)

function action_to_index(a::CartesianIndex{2})
    return ACTIONS[a]
end

function stack_exp(batch::Vector{Experience})
    batch_size = length(batch)
    h, w = size(batch[1][1])  # board dimensions

    # Preallocate arrays
    states_array      = Array{Float64}(undef, h, w, 1, batch_size)
    next_states_array = Array{Float64}(undef, h, w, 1, batch_size)
    actions_array     = Array{Int}(undef, batch_size)
    rewards_array     = Array{Float64}(undef, batch_size)
    done_array        = Array{Bool}(undef, batch_size)

    for i in 1:batch_size
        s, a, r, s_next, done = batch[i]

        states_array[:, :, 1, i]      .= Float32.(s)
        next_states_array[:, :, 1, i] .= Float32.(s_next)
        actions_array[i]  = action_to_index(a)
        rewards_array[i]  = Float32(r)
        done_array[i]     = done
    end

    return (states_array, actions_array, rewards_array, next_states_array, done_array)
end
#-----------------------------------------------------------------------------------------------------------
#batch comes out from sample in buffer.jl
function dqn_loss(model::DQNModel, stacked_batch::Tuple{Array{Float64}, Array{Float64}, Array{Int}, Array{Float64}, Array{Bool}}, game::SnakeGame)::Float64
    states, actions, rewards, next_states, dones = stacked_batch 
    total_loss = 0.0
    q_pred = model.q_net(states)     #(4, batch_size)
end


#start by acquiring experience until the buffer is full
game = SnakeGame()
rpb = ReplayBuffer()
model = DQNModel(game)

