#utils.jl
################################################## environment methods ###############################################
#get available actions: reverse is never an option
function available_actions(game::SnakeGame)
          all_actions = [CartesianIndex(-1,0), CartesianIndex(1,0), CartesianIndex(0,-1), CartesianIndex(0,1)]
          return [a for a in all_actions if a + game.prev_move != CartesianIndex(0,0)]
end

# Function to plot the game state
function plot_or_update!(game::SnakeGame, plt::Union{Plots.Plot, Missing, Nothing} = missing)
    h, w = size(game.state)
    img = fill(ARGB32(1, 1, 1, 1), h, w)    # White background

    for i in 1:h, j in 1:w
        if game.state[i, j] == -1               # Walls
            img[i, j] = ARGB32(0, 0, 0, 1)  # Black
        elseif game.state[i, j] == 1            # Snake
            img[i, j] = ARGB32(0, 1, 0, 1)  # Green
        elseif game.state[i, j] == 2            # Food
            img[i, j] = ARGB32(1, 0, 0, 1)  # Red
        end
    end
    
    if ismissing(plt) || isnothing(plt)
        return plot(img, framestyle = :none)
    else   
        plot!(plt, img)
        return plt
    end 
end

# Function to place new food, not really random: the seed is fixed at the beginning of a new episode.

function sample_food!(game::SnakeGame)

    # Find empty positions
    empty_positions = findall(==(0),game.state)
    
    if isempty(empty_positions)
        @info "no more empty positions"
        return  # No space left
    end
    
    food_pos = 0
    # Pick the first available food position
    for f in game.food_list
         if f in empty_positions
             food_pos = f
             idx = findfirst(==(f), game.food_list)
             if idx !== nothing
                 deleteat!(game.food_list, idx)
             end
             break
         end
    end

    # Update game state
    game.state[food_pos] = 2
    
    @debug "Placing food at $food_pos"
end 

#update state
function update_state!(game::SnakeGame)

    game.state[game.state .== 1] .= 0 #remove previous snake positions

    # Redraw snake, this also handle removing food pixel when needed
    for ci in game.snake
        game.state[ci] = 1
    end
    
end

#function to check collision, must be modified
function check_collision(game::SnakeGame) :: Bool
    head = game.snake[1]
    return game.state[head] == -1 ||  count(==(head), game.snake) > 1 ||  game.prev_move + game.direction == CartesianIndex(0,0)   #true for a collision false otherwise, first condition check collision with the wall, second and third ones collisions with the snake itself. TODO:last condition should be useless
end

#remove tail
function remove_tail!(game::SnakeGame)
    pop!(game.snake)
end

#function to grow maybe
function grow_maybe!(game::SnakeGame)
    new_head = game.snake[1] + game.direction
    pushfirst!(game.snake, new_head)

    # If food is eaten, place new food and increase score and reward
    if game.state[new_head] == 2  
        game.score += 1
        game.reward = game.eating_reward                               #immediate reward = 1 for eating food
        @debug "food in ($(new_head[1]),$(new_head[2])) eaten!" reward = game.reward
        sample_food!(game)
    else
        remove_tail!(game)  # Normal move (no food)
        game.reward = game.male_di_vivere                              #small penalty for surviving without eating anything
        @debug "no food is eaten!" reward = game.reward
    end
end

#move wrapper, the snakes grow, if he hits the wall or himself he loses, if he does not run into food, I remove the tail 

function move_wrapper!(game::SnakeGame)
    grow_maybe!(game)  # Add new head
    
    if check_collision(game) 
        game.lost = true 
        game.reward = game.suicide_penalty                             #immediate reward = -1 if he looses
    end

    update_state!(game)
    game.prev_move = game.direction
end

#get_step seems to be broken, saves the same state twice and execute the action twice
function get_step(game::SnakeGame, action::CartesianIndex{2})::Experience
         
          state = deepcopy(game.state)
          av_actions = available_actions(game)
          game.direction = action
          move_wrapper!(game)
          reward = deepcopy(game.reward)
          next_state = deepcopy(game.state)
          if game.lost
             return (state, action, reward, next_state, true, av_actions)
          else 
             return (state, action, reward, next_state, false, av_actions)
          end
end

#####################################################buffer methods#################################################################

#definition length of a buffer
Base.length(rpb::ReplayBuffer) = length(rpb.buffer)

function store!(rpb::ReplayBuffer, exp::Experience)
          if length(rpb) < rpb.capacity
              push!(rpb.buffer, exp)
          else
             rpb.buffer[rpb.position] = exp
             rpb.position += 1 
          end
          if rpb.position > rpb.capacity
              rpb.position = 1
          end
end

#I think sampling without replacement is faster because I do not want to repeat the initial state-actions for too many times.
function sample(rpb::ReplayBuffer)::Vector{Experience}
          batch_size = rpb.batch_size
          if length(rpb) < batch_size
              return StatsBase.sample(rpb.buffer_rng, rpb.buffer, length(rpb); replace = false)     
          else
              return StatsBase.sample(rpb.buffer_rng, rpb.buffer, batch_size; replace = false)
          end 
end

function isfull(rpb::ReplayBuffer)
          return rpb.position == rpb.capacity
end

function isready(rpb::ReplayBuffer)
          batch_size = rpb.batch_size
          return length(rpb) >= batch_size
end

function fill_buffer!(rpb::ReplayBuffer, model::DQNModel; epsilon::Union{Float32,Missing} = missing)
         @info "############################## FILLING THE BUFFER ###############################"
         game = SnakeGame()
         eps = ismissing(epsilon) ? 1.0f : epsilon
         while !isfull(rpb)
               # epsilon-greedy policy
               action = epsilon_greedy(game, model, eps)
               exp = get_step(game, action)
               store!(rpb, exp)
               if game.lost game = SnakeGame() end 
         end 
         @info "##############################  BUFFER FULL ######################################"
end 

function empty_buffer!(rpb::ReplayBuffer)
           rpb.buffer = Vector{Experience}(undef, 0)
           rpb.position = 1
end 

##################################################model methods################################################################

function epsilon_greedy(game::SnakeGame, model::DQNModel, epsilon::Float32; temp_model::Union{Chain,Nothing}=nothing)::CartesianIndex{2}

          av_actions = available_actions(game)
          
          state = reshape(game.state, game.state_size, game.state_size, 1, 1)
          
          if Float32(only(rand(model.model_rng,1))) < epsilon
              act = rand(model.model_rng, av_actions)
          else
          
             exp_rewards = isnothing(temp_model) ? model.q_net(state) : temp_model(state)
             max_idx = argmax(exp_rewards)
             act = av_actions[max_idx]
             #@debug "best action selected!" expected_rewards = exp_rewards max_idx = max_idx
          end

          return act
end

function update_target_net!(model::DQNModel)
          flat_params, reconstructor = Flux.destructure(model.q_net)
          model.t_net = reconstructor(flat_params)
end

function save_model(model::DQNModel, name::String)
          
          path = "./models/"
          if !isdir(path) mkpath(path) end
          @save path * name * ".bson" model
end

function load_model(dir::String; temp::Union{Bool,Missing}=missing)::Union{DQNModel,Chain}
    if ismissing(temp)||temp == false
       model = nothing
       @load dir model
       return model
    else
       temp_model = nothing 
       @load dir temp_model
       return temp_model
    end
end

###########################batch manipulation functions#################################################################################   
function stack_exp(batch::Vector{Experience})
    batch_size = length(batch)
    h, w = size(batch[1][1])  # board dimensions

    # Preallocate arrays
    states_array      = Array{Float32}(undef, h, w, 1, batch_size)
    next_states_array = Array{Float32}(undef, h, w, 1, batch_size)
    actions_array     = Array{Int}(undef, batch_size)
    rewards_array     = Array{Float32}(undef, batch_size)
    done_array        = Array{Bool}(undef, batch_size)
    av_acts_array     = Array{Vector{CartesianIndex{2}}}(undef, batch_size)    #for debugging
    a_array           = Array{CartesianIndex{2}}(undef, batch_size)            #for debugging

    for i in 1:batch_size
        s, a, r, s_next, done, av_acts = batch[i]

        states_array[:, :, 1, i]      .= Float32.(s)
        next_states_array[:, :, 1, i] .= Float32.(s_next)
        actions_array[i]  = findfirst(isequal(a), av_acts)
        rewards_array[i]  = Float32(r)
        done_array[i]     = done
        av_acts_array[i]  = av_acts
        a_array[i]        = a 
    end

    return (states_array, actions_array, rewards_array, next_states_array, done_array, av_acts_array, a_array) #better to save also av_acts and a to debug further
end

############################################################visualization functions##############################################
#log infos

function log_hyperparameters(trainer::Trainer)
    game = trainer.game
    model = trainer.model
    buffer = trainer.buffer

    hyperparams = Dict(
        # Environment rewards
        :eating_reward => game.eating_reward,
        :suicide_penalty => game.suicide_penalty,
        :male_di_vivere => game.male_di_vivere,

        # Environment settings
        :state_size => game.state_size,
        :discount => game.discount,

        # Training
        :epsilon_start => trainer.epsilon,
        :epsilon_end => trainer.epsilon_end,
        :epsilon_decay => trainer.decay,
        :n_batches => trainer.n_batches,
        :target_update_rate => trainer.target_update_rate,
        :replay_buffer_capacity => buffer.capacity,
        :replay_buffer_batch_size => buffer.batch_size,

        # Optimizer
        :learning_rate => model.opt.eta
    )

    @info "=== Hyperparameters ==="
    for (k, v) in hyperparams
        @info string(k) * " => " * string(v)
    end

    return nothing
end


function plot_loss(tr::Trainer; size::Int = 100, save_name::String)
    batch_size = tr.buffer.batch_size
    n_batches = tr.n_batches

    # X: number of experience samples
    x = [i * batch_size for i in 0:length(tr.losses)-1]
    y = tr.losses

    # Plot original loss
    pl = plot(x, y, label = "Loss", lw = 2, lc = :blue)
    xlabel!("Experience Samples")
    ylabel!("Loss")

    # Add moving average if requested
    if length(y) ≥ size
        z = [mean(y[i-size+1:i]) for i in size:length(y)]
        x_avg = x[size:end]
        plot!(x_avg, z, label = "Moving Avg", lw = 2, lc = :red)
    end
       
    path = "./plots/"
    if !isdir(path) mkpath(path) end
    savefig(pl, path * save_name * ".png")

end

function play_game(tr::Trainer, gif_name::String)
          plt = nothing
          game = SnakeGame()
          model = tr.model
          plt = plot_or_update!(game)
                    
          break_next = false
         
          anim = @animate for i in 1:400                                    #I want to exit the loop when game.lost
             if i > 1 && break_next == false
                 game.direction = epsilon_greedy(game, model, 0.0f0)        #always the best action
                 move_wrapper!(game)
             end
             
             if break_next == true
                break
             end 
             
             plot_or_update!(game, plt)
             
             if game.lost
                 @printf "Collision!! Final score %d \n" game.score
                 break_next = true
             end
         end
         
         path = "./trainer_gifs/"
         if !isdir(path) mkpath(path) end
         gif(anim, path*gif_name*".gif", fps=1)
         return nothing    
end

function state_to_img!(state::Matrix{Int})
           h, w = size(state)
           img = fill(ARGB32(1, 1, 1, 1), h, w)    # White background

           for i in 1:h, j in 1:w
               if state[i, j] == -1               # Walls
                   img[i, j] = ARGB32(0, 0, 0, 1)  # Black
               elseif state[i, j] == 1            # Snake
                   img[i, j] = ARGB32(0, 1, 0, 1)  # Green
               elseif state[i, j] == 2            # Food
                    img[i, j] = ARGB32(1, 0, 0, 1)  # Red
               end
            end

         return img
end 

#visualize a game sampled from the buffer, something in the storing of experience in the buffer is wrong
function sample_game(rpb::ReplayBuffer; gif_name::String)
          
          #sampling the game
          game_states = []
          start_idx = rand(1:length(rpb) - Int((floor(30/100 * length(rpb))))) #I want a diffent game everytime I call this function 
          copy_idx = 0
          done = false
          
        
          for i in start_idx:length(rpb)
               if rpb.buffer[i][5] 
                   copy_idx = i + 1
                   break
               end
          end
          
          
          while !done
                example = rpb.buffer[copy_idx]
                push!(game_states, example[1], example[4])   #append state and next_state
                copy_idx += 1
                done = example[end]
          end
          
          unique_states = unique(game_states)
          frames = state_to_img!.(unique_states)
          
          
          #plotting the game
          anim = @animate for i in 1:length(frames)
                  plot(frames[i], framestyle = :none)
          end
          
          path = "./buffer_gifs/"
          if !isdir(path) mkpath(path) end
          gif(anim, path*gif_name*".gif", fps=1)
          return game_states
end

#plot average rewards
function plot_avg_rewards(tr::Trainer; size::Int=100, save_name::String)
          x = [i for i in 0:length(tr.episode_rewards)-1]
          y = tr.episode_rewards
          if length(y) ≥ size
              z = [mean(y[i-size+1:i]) for i in size:length(y)]
              x_avg = x[size:end]
              pl = plot(x_avg, z, label = "Moving Avg", lw = 2, lc = :red)
              xlabel!("episodes")
              ylabel!("average reward over $size episodes")
          end
          path = "./episode_rewards/"
          if !isdir(path) mkpath(path) end
          savefig(pl, path*save_name*".png")
          return nothing    
end     

function plot_and_play(tr::Trainer; size::Int=100, save_name::String)
          plot_loss(tr; size = size, save_name = save_name)
          plot_avg_rewards(tr; size = size, save_name = save_name)
          play_game(tr, save_name)
end 
#############################################trainer methods#####################################################################

#method fill the buffer for the trainer, useful to debug
function fill_buffer!(tr::Trainer)
         @info "############################## FILLING THE BUFFER ###############################"
         game = SnakeGame()
         
         while !isfull(tr.buffer)
               # epsilon-greedy policy
               action = epsilon_greedy(game, tr.model, tr.epsilon)
               exp = get_step(game, action)
               store!(tr.buffer, exp)
               if game.lost game = SnakeGame() end               
         end 
         @info "##############################  BUFFER FULL ######################################"
end 

function track_loss!(tr::Trainer, item)
          push!(tr.losses, item)
end

#loading the model, from the Trainer class
function load_model!(tr::Trainer, dir::String)
          tr.model = load_model(dir)
end

function save_trainer(tr::Trainer, name::String)
          path = "./trainers/"
          if !isdir(path) mkpath(path) end
          @save path * name * ".bson" tr
end

function load_trainer(dir::String)::Trainer
          tr = nothing
          @load dir tr
          return tr
end

function train!(tr::Trainer; trainer_name::String)
          
          #defining variables
          if tr.save log_hyperparameters(tr) end
      
          n_batches = tr.n_batches
          game = tr.game
          target_update_rate = tr.target_update_rate
          
          fill_buffer!(tr.buffer, tr.model)
          opt_state = Flux.setup(tr.model.opt, tr.model.q_net)
          episode_reward = 0.0f0
          nb = 0
          
          @info "############################## START TRAINING ###############################"
          while nb <= n_batches
                 
                 action = epsilon_greedy(game, tr.model, tr.epsilon)
                 experience = get_step(game, action)
                 
                 episode_reward += experience[3]
                 store!(tr.buffer, experience)
                 
                 batch = sample(tr.buffer)
                 states, actions, rewards, next_states, dones, av_actions, a_array = stack_exp(batch)
                 
                 q_pred = tr.model.q_net(states)                                            # (n_actions, batch_size)
    		 q_pred_selected = [q_pred[a, i] for (i, a) in enumerate(actions)]
    		 q_pred_selected = reshape(q_pred_selected, :)                              # (batch_size,)
    		 q_next = tr.model.t_net(next_states)
    		 max_next_q = dropdims(maximum(q_next, dims = 1), dims = 1)                 # (batch_size,)
		 q_target = @. rewards + game.discount * max_next_q * (1 - dones)
		 
		 if nb % 100000 == 0                        
		     @debug "######################### BATCH $nb ###################################"
		     @debug "Actions_selected:" actions = actions
		     @debug "Q_pred:" q_pred
		     @debug "Shape Q-pred" q_pred_shape = size(q_pred)
		     @debug "Shape Q-pred-selected" q_pred_selected_shape = size(q_pred_selected)
		     @debug "Shape Q-target" q_target_shape = size(q_target)
		     @debug "Q_pred_selected:" q_pred_selected = q_pred_selected
		     @debug "Action_selected_in_cartesian_index"  a_array = a_array
		     @debug "States" states = states
		     @debug "Next_states" next_states = next_states
		 
                     #@debug "Max_next_q:" max_next_q =max_next_q
                     #@debug "Q_target:" q_target=q_target
		 end
		 function loss_fun(z)
		           q_pred_selected = [z[a, i] for (i, a) in enumerate(actions)]
		           q_pred_selected = reshape(q_pred_selected, :)
		           return Flux.huber_loss(q_pred_selected, q_target; agg = mean)
		 end
		 
		 #doing the update
		 grads = Flux.gradient(tr.model.q_net) do m
		       q_pred = m(states)
                       loss_fun(q_pred)
                 end

                 Flux.update!(opt_state, tr.model.q_net, only(grads))
                 if isnothing(only(grads)) @warn "Network has not been updated" end
		 
		 if nb % target_update_rate == 0 
		     update_target_net!(tr.model) 
		     @info "Batch $nb | Target network updated." 
		 end
		 
		 if game.lost
		     push!(tr.episode_rewards, episode_reward)
		     episode_reward = 0.0f0
		     game = SnakeGame()
		     #@info "Batch $nb | Game reset due to loss." 
		 end
		 
		 if nb % 5 == 0 
		     @printf "%d / %d -- loss %.3f \n" nb n_batches Flux.huber_loss(q_pred_selected, q_target)
		 end
		 		 
		 track_loss!(tr, Flux.huber_loss(q_pred_selected, q_target))
		 tr.epsilon = max(tr.epsilon - tr.decay, tr.epsilon_end)                            #linear epsilon decay
		 nb += 1
          end 
                  
          @info "############################## END TRAINING ###############################"
          
          if tr.save 
              
              plot_and_play(tr; save_name = trainer_name)
              empty_buffer!(tr.buffer)                      #freeing up space
              save_trainer(tr, trainer_name) 
              @info "Trainer state saved to $trainer_name" 
          end  
end 


###########################################LaplaceTrainer methods#####################################################
######################################################################################################################
#Here is how LA will work.
#1.Detect plateau fitting a line over 500 samples
#2.At this point iniziatizing the first and the second moments
#3.Start updating them and adding column to D
#4.In the meanwhile q_net is of course training and filling the buffer.
#5.After 100 iterations of the loop D is fool and I am ready to sample a new model
#6.Now, I am training always q_net but gathering experience using Laplace
#7.After 500 other batches I sample another model.
#8.Continue till average reward is increasing again and then restart using E-greedy.   
######################################################################################################################
function track_loss!(tr::LaplaceTrainer, item)
          push!(tr.losses, item)
end

function save_trainer(tr::LaplaceTrainer, name::String)
          path = "./laplace_trainers/"
          if !isdir(path) mkpath(path) end
          @save path * name * ".bson" tr
end

function load_la_trainer(dir::String)::LaplaceTrainer
          tr = nothing
          @load dir tr
          return tr
end

#check if there is a plateau, TODO:modify followig the next function
function check_plateau(tr::LaplaceTrainer; window::Int= 500, batch_number = 1)::Bool
         #skip the first plateau (10 000 samples)
         
         if batch_number < 10000
             return false 
         end 
         
         #fitting a line and finding the slope.
         y = tr.episode_rewards[end - window : end] 
         N = length(y)
         x = collect(1:N)
         X = hcat(ones(N), x)
         coeffs = X \ y                  #least mean square
         slope = coeffs[2]
         
         @info "slope is $slope"
         return slope < 0.1              #slope almost flat I apply Laplace
end

function check_plateau(tr::Trainer; window::Int=2000, fig::Union{Missing, Bool}=false)::Bool

          
          #Here I do not skip anything because I have already trained the model and reached a plateau
          len_rewards = length(tr.episode_rewards)
          y = tr.episode_rewards[end - window : end] 
          
          if minimum(y) < -10 return false end #maybe this solves

          N = length(y)
          x = collect(len_rewards - window : len_rewards)
          X = hcat(ones(N), x)
          coeffs = X \ y                  #least mean square
          slope = coeffs[2]
          
          if fig
              pl = plot(tr.episode_rewards, label = "Episode rewards", lw = 2, lc = :red)
              plot!(pl, x, @.(coeffs[2]*x + coeffs[2]), lw = 2, lc = :navy)
              xlabel!("episodes")
              ylabel!("rewards")
              display(pl)
          end
          
          println("slope is", slope)
          return -0.01 < slope < 0.01              #slope almost flat I apply Laplace
end

Base.length(tr::LaplaceTrainer) = size(tr.deviation_matrix, 2)    #returns the number of columns of the deviation matrix

function add_to_deviation_mat!(tr::LaplaceTrainer; dev_col::Vector{Float64})
          if length(tr) < tr.capacity
              tr.deviation_matrix = hcat(tr.deviation_matrix, dev_col)
          else
             tr.deviation_matrix[:, tr.position] = dev_col
             tr.position += 1 
          end
          if tr.position > tr.capacity
              tr.position = 1
          end          
end

function compute_Gamma_diag(var_SWA::Vector{Float64})
         diag_elements = var_SWA
         if minimum(diag_elements) < 0
            @warn "Gamma_diag has negative element, whose value is $(minimum(diag_elements))"
            diag_elements = abs.(diag_elements)
         end
         return Diagonal(diag_elements)
end

function sample_model(theta_SWA::Array{Float64}, theta_2_SWA::Array{Float64}, D::Matrix{Float64}, restructure)
          
          Gamma_diag = compute_Gamma_diag(theta_SWA, theta_2_SWA)
          d = length(theta_SWA)                                               #total size of the model
          K = size(D)[2] == 100 ? 100 : @error "Size of D is not 100, but $size(D)[2]"  
          nd = MvNormal(fill(.0,d), I)   
          nK = MvNormal(fill(.0,K), I)
          
          z1 = rand(nd)
          z2 = rand(nK) 
          
          w = theta_SWA + 1/sqrt(2) * sqrt.(Gamma_diag) * z1 + 1/sqrt(2*(K - 1)) * D * z2
          return restructure(w)    
end 

function sample_model_with_memory_mapping(theta_SWA::Vector{Float64}, var_SWA::Vector{Float64}, restructure; D::AbstractMatrix)
   
           Gamma_diag = compute_Gamma_diag(var_SWA)
          
           d,K = size(D)
        
           nd = MvNormal(fill(.0,d), I)   
           nK = MvNormal(fill(.0,K), I)
          
           z1 = rand(nd)
           z2 = rand(nK)
           
           max_block = 100
           Dz2 = zeros(Float64, d)
           n_blocks = K ÷  max_block
           
           for i in 1:n_blocks
                from = (i-1) * max_block + 1
                to = i * max_block
                z2_block = z2[from:to]
                D_block = D[:,from:to]
                Dz2 += D_block * z2_block
           end  
           
           if K % max_block != 0
              Dz2 += D[:, n_blocks*max_block + 1 : K] * z2[n_blocks*max_block + 1 : K]
           end
           
           w = theta_SWA + 1/sqrt(2) * sqrt.(Gamma_diag) * z1 + 1/sqrt(2*(K - 1)) * Dz2
           return restructure(w)
end 

#algorithm to compute efficiently average and variance
function welford_update(count::Int, aggregate::Tuple{Vector{Float64}}, new_value)
          
          mean, m2 = aggregate
          count += 1
          delta = @.(new_value - mean)
          mean += delta / count
          delta2 = @.(new_value - mean)
          m2 += @.(delta * delta2)
          return count, mean, m2
end

function welford_finalize(count::Int, aggregate::Tuple{Vector{Float64}})
          
          mean, m2 = aggregate
          if count < 2 
             throw("Count must be greater than or equal to 2")
          else
             variance = m2/(count - 1)
          end
          return mean, variance
end

#TODO:to be modified following resume_training! function
function train!(tr::LaplaceTrainer; trainer_name::String)
          
          #defining variables
          if tr.save log_hyperparameters(tr) end
      
          n_batches = tr.n_batches
          game = tr.game
          target_update_rate = tr.target_update_rate
          param_count = length(Flux.destructure(tr.model.q_net)[2])
          
          fill_buffer!(tr.buffer, tr.model)
          opt_state = Flux.setup(tr.model.opt, tr.model.q_net)
          episode_reward = 0.0f0
          treshold = tr.capacity                        #treshold, basically the number of columns of the deviation_matrix
          laplace_counter = 0
          
           #initializing variables
          theta_SWA, re = Flux.destructure(tr.model.q_net)
          theta_2_SWA = (theta_SWA).^2
          temp_model = nothing
          
          nb = 0
          
          @info "############################## START TRAINING ###############################"
          while nb <= n_batches
                 
                 #deciding whether I am in Laplace regime or not
                 tr.laplace = check_plateau(tr; window = 500, batch_number = nb)
                 
                 #if Laplace regime is starting initialize first and second moments of the weights
                 if tr.laplace&&laplace_counter == 0
                     @info "starting LA"                    
                     theta_SWA, re = Flux.destructure(tr.model.q_net)
                     theta_2_SWA = (theta_SWA).^2
                 end  
                 
                 #If Laplace regime and D matrix is already full I can start sample actions from posterio, otherwise epsilon-greedy
                 if tr.laplace&&laplace_counter > treshold
                     
                     if (laplace_counter - treshold) % 500 == 0 
                         @info "sampling a model for LA"
                         temp_model = sample_model(theta_SWA, theta_2_SWA, tr.deviation_matrix, re)  
                     end   
                                        
                     action = epsilon_greedy(game, temp_model, 0.0f0)          
                 else
                     
                     #If there has been a Laplace regime but now is finished I reset Laplace variables
                     if tr.laplace&&laplace_counter > 0
                         
                         @info "aborting LA"
                         #reset values
                         laplace_counter = 0                     
                         tr.deviation_matrix = deepcopy(Matrix{Float32}(undef, param_count, 0))
                     end 
                       
                     action = epsilon_greedy(game, tr.model, tr.epsilon)
                 end
                 
                 experience = get_step(game, action)
                 
                 episode_reward += experience[3]
                 store!(tr.buffer, experience)
                 
                 batch = sample(tr.buffer)
                 states, actions, rewards, next_states, dones, av_actions, a_array = stack_exp(batch)
                 
                 q_pred = tr.model.q_net(states)                                            # (n_actions, batch_size)
    		 q_pred_selected = [q_pred[a, i] for (i, a) in enumerate(actions)]
    		 q_pred_selected = reshape(q_pred_selected, :)                              # (batch_size,)
    		 q_next = tr.model.t_net(next_states)
    		 max_next_q = dropdims(maximum(q_next, dims = 1), dims = 1)                 # (batch_size,)
		 q_target = @. rewards + game.discount * max_next_q * (1 - dones)
		 
		 if nb % 10000000 == 0          #this works
		     @debug "######################### BATCH $nb ###################################"
		     @debug "Actions_selected:" actions = actions
		     @debug "Q_pred:" q_pred
		     @debug "Shape Q-pred" q_pred_shape = size(q_pred)
		     @debug "Shape Q-pred-selected" q_pred_selected_shape = size(q_pred_selected)
		     @debug "Shape Q-target" q_target_shape = size(q_target)
		     @debug "Q_pred_selected:" q_pred_selected = q_pred_selected
		     @debug "Action_selected_in_cartesian_index"  a_array = a_array
		     @debug "States" states = states
		     @debug "Next_states" next_states = next_states
		 
		 end
		 function loss_fun(z)
		           q_pred_selected = [z[a, i] for (i, a) in enumerate(actions)]
		           q_pred_selected = reshape(q_pred_selected, :)
		           return Flux.huber_loss(q_pred_selected, q_target; agg = mean)
		 end
		 
		 #doing the update
		 grads = Flux.gradient(tr.model.q_net) do m
		       q_pred = m(states)
                       loss_fun(q_pred)
                 end

                 Flux.update!(opt_state, tr.model.q_net, only(grads))
                 if isnothing(only(grads)) @warn "Network has not been updated" end
                 
                 #if Laplace Regime update momenta and add column to D
                 if tr.laplace
                     theta, _ = Flux.destructure(tr.model.q_net)
                     theta_SWA = @.(laplace_counter * theta_SWA + theta)/(laplace_counter + 1) 
                     theta_2_SWA = @.(laplace_counter * theta_2_SWA + theta^2)/(laplace_counter + 1) 
                     dD = theta - theta_SWA
                     add_to_deviation_mat!(tr; dev_col=dD)             
                 end
		 
		 if nb % target_update_rate == 0 
		     update_target_net!(tr.model) 
		     @info "Batch $nb | Target network updated." 
		 end
		 
		 if game.lost
		     push!(tr.episode_rewards, episode_reward)
		     episode_reward = 0.0f0
		     game = SnakeGame()
		     @info "Batch $nb | Game reset due to loss." 
		 end
		 
		 if nb % 5 == 0 
		     @printf "%d / %d -- loss %.3f \n" nb n_batches Flux.huber_loss(q_pred_selected, q_target)
		 end
		 		 
		 track_loss!(tr, Flux.huber_loss(q_pred_selected, q_target))
		 
		 #If I am in Laplace regime update counter, if I am not decay epsilon
		 if tr.laplace
		    laplace_counter += 1 	
		 else
		      tr.epsilon = max(tr.epsilon - tr.decay, tr.epsilon_end)                            #linear epsilon decay
		 end
		
		 nb += 1
          end 
                  
          @info "############################## END TRAINING ###############################"
          
          if tr.save 
              
              plot_and_play(tr; save_name = trainer_name)
              empty_buffer!(tr.buffer)                      #freeing up space
              save_trainer(tr, trainer_name) 
              @info "Trainer state saved to $trainer_name" 
          end  
end
 
#does not work, trying to use mse as loss, maybe is more regular
#Resume training function, accepts a Trainer and trains it has a LaplaceTrainer.
function resume_training!(;n_batches::Int=100000, trainer_path::String, la_trainer_name::String) #trainer_path for loading, la_trainer_name for saving
          
          tr = load_trainer(trainer_path)
          #defining variables
          if tr.save log_hyperparameters(tr) end
      
          #n_batches = tr.n_batches
          game = tr.game
          target_update_rate = tr.target_update_rate
          param_count = length(Flux.destructure(tr.model.q_net)[2])
          
          fill_buffer!(tr.buffer, tr.model)
          opt_state = Flux.setup(tr.model.opt, tr.model.q_net)
          
          #defining the deviation matrix and the Laplace boolean variable
         
          capacity = 1000 
          position = 1
          laplace = false
          deviation_file_path = "./deviation_matrix.bin"
          deviation_file = nothing
          deviation_matrix = nothing

          
          #initializing variables
          _, re = Flux.destructure(tr.model.q_net)  #here is important to use more precision
          theta_SWA = zeros(param_count)            #initialization for welford's algorithm
          m2 = zeros(param_count)
          temp_model = nothing
          
          episode_reward = 0.0f0
          treshold = capacity                        #treshold, basically the number of columns of the deviation_matrix
          laplace_counter = 0
          
          nb = 0
          
          @info "############################## START TRAINING ###############################"
          while nb <= n_batches
                 
                 #deciding whether I am in Laplace regime or not
                 laplace = check_plateau(tr; window = 2000)
                 
                 #if Laplace regime is starting initialize first and second moments of the weights
                 if laplace&&laplace_counter == 0
                     @info "starting LA" 
                      
                     # Create or open the file
                     isfile(deviation_file_path) || open(deviation_file_path, "w") do io
                              # Reserve space: each Float64 takes 8 bytes
                              write(io, zeros(Float64, param_count * capacity))
                     end

                     # Open for reading/writing + mmap
                     deviation_file = open(deviation_file_path, "r+")
                     deviation_matrix = mmap(deviation_file, Matrix{Float64}, (param_count, capacity))
                                       
                     theta_SWA, re = Flux.destructure(tr.model.q_net)
                     theta_SWA = Float64.(theta_SWA)
                     theta_2_SWA = (theta_SWA).^2
                 end  
                 
                 #If Laplace regime and D matrix is already full I can start sample actions from posterior, otherwise epsilon-greedy
                 if laplace&&laplace_counter >= treshold
                     
                     if (laplace_counter - treshold) % 500 == 0 
                         @info "sampling a model for LA" 
                         theta_SWA, var_SWA = welford_finalize(laplace_counter, [theta_SWA, m2])
                         temp_model = sample_model_with_memory_mapping(theta_SWA, theta_2_SWA, re; D = deviation_matrix)
                         
                         #saving temp models for further analysis (in particular I need to inspect the buffer later)
                         n = div((laplace_counter - treshold), 500)
                         path = "./temp_models/"
          	         if !isdir(path) mkpath(path) end
                         @save path * la_trainer_name * "_temp_model_$n" * ".bson" temp_model
                          
                     end   
                                        
                     action = epsilon_greedy(game, tr.model, 0.0f0; temp_model = temp_model)          
                 else
                     
                     if !laplace&&laplace_counter > 0
                         
                         @info "aborting LA"
                         #reset values
                         if deviation_file !== nothing && isopen(deviation_file)
                            close(deviation_file)
                         end
                         rm("deviation_matrix.bin"; force = true)
                         laplace_counter = 0                                              
                     end 
                       
                     action = epsilon_greedy(game, tr.model, tr.epsilon)
                 end
                 
                 experience = get_step(game, action)
                 
                 episode_reward += experience[3]
                 store!(tr.buffer, experience)
                 
                 batch = sample(tr.buffer)
                 states, actions, rewards, next_states, dones, av_actions, a_array = stack_exp(batch)
                 
                 q_pred = tr.model.q_net(states)                                            # (n_actions, batch_size)
    		 q_pred_selected = [q_pred[a, i] for (i, a) in enumerate(actions)]
    		 q_pred_selected = reshape(q_pred_selected, :)                              # (batch_size,)
    		 q_next = tr.model.t_net(next_states)
    		 max_next_q = dropdims(maximum(q_next, dims = 1), dims = 1)                 # (batch_size,)
		 q_target = @. rewards + game.discount * max_next_q * (1 - dones)
		 
		 function loss_fun(z)
		           q_pred_selected = [z[a, i] for (i, a) in enumerate(actions)]
		           q_pred_selected = reshape(q_pred_selected, :)
		           return Flux.mse(q_pred_selected, q_target; agg = mean)
		 end
		 
		 #doing the update
		 grads = Flux.gradient(tr.model.q_net) do m
		       q_pred = m(states)
                       loss_fun(q_pred)
                 end

                 Flux.update!(opt_state, tr.model.q_net, only(grads))
                 if isnothing(only(grads)) @warn "Network has not been updated" end
                 
                 #if Laplace Regime update momenta and add column to D
                 if laplace
                     theta, _ = Flux.destructure(tr.model.q_net)
                     theta = Float64.(theta)
                     laplace_counter, theta_SWA, m2  = welford_update(laplace_counter, (theta_SWA, m2), theta)
                     dD = theta - theta_SWA  
                    
                     deviation_matrix[:, position] = dD             
                     position = position == capacity ? 1 : position + 1               
                 end
		 
		 if nb % target_update_rate == 0 
		     update_target_net!(tr.model) 
		     @info "Batch $nb | Target network updated." 
		 end
		 
		 if game.lost
		     push!(tr.episode_rewards, episode_reward)
		     episode_reward = 0.0f0
		     game = SnakeGame()
		     @info "Batch $nb | Game reset due to loss." 
		 end
		 
		 if nb % 5 == 0 
		     @printf "%d / %d -- loss %.3f \n" nb n_batches Flux.mse(q_pred_selected, q_target)
		 end
		 		 
		 track_loss!(tr, Flux.mse(q_pred_selected, q_target))
		 
		 #If I am in Laplace regime update counter, if I am not decay epsilon TODO:update the count in welford_update
		 if laplace
		    laplace_counter += 1 	
		 else
		      tr.epsilon = max(tr.epsilon - tr.decay, tr.epsilon_end)                            #linear epsilon decay
		 end
		
		 nb += 1
          end 
          
          if deviation_file !== nothing && isopen(deviation_file)
             close(deviation_file)
          end
          
          if isfile(deviation_file_path)
             rm(deviation_file_path; force = true)
          end

                  
          @info "############################## END TRAINING ###############################"
          
          if tr.save 
              
              plot_and_play(tr; save_name = la_trainer_name)
              empty_buffer!(tr.buffer)                      #freeing up space
              save_trainer(tr, la_trainer_name) 
              @info "Trainer state saved to $la_trainer_name" 
          end

end

#function to fill the buffer using a temp_model. Useful for debugging
function temp_model_fills_buffer(temp_name::String)
         path = "./temp_models/"
         full_path = path * temp_name
         gm = SnakeGame()
         md = DQNModel(gm)
         md.q_net = load_model(full_path; temp=true)
         bf = ReplayBuffer(10000)
         fill_buffer!(bf, md; epsilon=0.0f0)
         return bf
end 
