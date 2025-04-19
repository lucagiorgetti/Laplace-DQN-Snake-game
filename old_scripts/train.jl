#train loop

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
    states_array      = Array{Float32}(undef, h, w, 1, batch_size)
    next_states_array = Array{Float32}(undef, h, w, 1, batch_size)
    actions_array     = Array{Int}(undef, batch_size)
    rewards_array     = Array{Float32}(undef, batch_size)
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

function train!(model::DQNModel, rpb::ReplayBuffer, n_batches::Int, target_update_rate::Int, epsilon::Float32)
          fill_buffer!(rpb, model)
          opt_state = Flux.setup(model.opt, model.q_net)
          nb = 0
          
          game = SnakeGame()
          while nb < n_batches
                 action = epsilon_greedy(game, model, epsilon)
                 experience = get_step(game, action)
                 store!(rpb, experience)
                 
                 batch = sample(rpb)
                 states, actions, rewards, next_states, dones = stack_exp(batch)
                 q_pred = model.q_net(states)                                               # (n_actions, batch_size)
    		 q_pred_selected = [q_pred[a, i] for (i, a) in enumerate(actions)]
    		 q_pred_selected = reshape(q_pred_selected, :)                              # (batch_size,)
    		 q_next = model.t_net(next_states)
    		 max_next_q = dropdims(maximum(q_next, dims = 1), dims = 1)                 # (batch_size,)
		 q_target = @. rewards + game.discount * max_next_q * (1 - dones)
		 
		 grads = Flux.gradient(model.q_net) do m
		     Flux.huber_loss(q_pred_selected, q_target)
		 end
		 Flux.update!(opt_state, model.q_net, grads[1])
		 
		 if nb % target_update_rate == 0 update_target_net!(model) end
		 if game.lost game = SnakeGame() end
		 if nb % 5 == 0 @printf "%d / %d -- loss %.3f \n" nb n_batches Flux.huber_loss(q_pred_selected, q_target) end
		 nb += 1
          end   
end 

