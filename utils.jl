#utils.jl
#####################################################################################################################
#now a state is composed by two frames instead of 1, now the role of the old state is supplied by the field board
######################################################################################################################
################################################## environment methods ###############################################
#get available actions: reverse is never an option
function available_actions(game::SnakeGame)
          all_actions = [CartesianIndex(-1,0), CartesianIndex(1,0), CartesianIndex(0,-1), CartesianIndex(0,1)]
          return [a for a in all_actions if a + game.prev_dir != CartesianIndex(0,0)]
end

# Function to plot the game, later on visualizatio functions
function sample_food!(game::SnakeGame)

    # Find empty positions
    empty_positions = findall(==(0),game.board)
    
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
    game.board[food_pos] = 2
    
    #@debug "Placing food at $food_pos"               #also printed in virtual step
end 

#update board
function update_board!(game::SnakeGame)

    game.board[game.board .== 1] .= 0 #remove previous snake positions

    # Redraw snake, this also handle removing food pixel when needed
    for ci in game.snake
        game.board[ci] = 1
    end
    
end

#function to check collision
function check_collision(game::SnakeGame) :: Bool
    head = game.snake[1]
    return game.board[head] == -1 ||  count(==(head), game.snake) > 1 ||  game.prev_dir + game.direction == CartesianIndex(0,0) #true for a collision false otherwise, first condition check collision with the wall, second and third ones collisions with the snake itself. Last condition has been added only for play_snake.jl
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
    if game.board[new_head] == 2  
        game.score += 1
        game.reward = game.eating_reward                                               #immediate reward = 1 for eating food
        #@debug "food in ($(new_head[1]),$(new_head[2])) eaten!" reward = game.reward  #the problem is that this is printed also in virtual_step
        sample_food!(game)
    else
        remove_tail!(game)  # Normal move (no food)
        game.reward = game.male_di_vivere                                              #small penalty for surviving without eating anything
       # @debug "no food is eaten!" reward = game.reward
    end
end

#move wrapper, the snakes grow, if he hits the wall or himself he loses, if he does not run into food, I remove the tail 

function move_wrapper!(game::SnakeGame)
    grow_maybe!(game)  # Add new head
    
    if check_collision(game)||length(game.board_history) > 500         #control on the length of the snake. It should                 
                                                                       #be only needed in play_best_game
        game.lost = true 
        game.reward = game.suicide_penalty                             #immediate reward = -1 if he looses
    end

    update_board!(game)
    game.prev_dir = game.direction
end


#step
function step!(game::SnakeGame, action::CartesianIndex{2})
          
          game.direction = action
          move_wrapper!(game)
          
          push!(game.board_history, deepcopy(game.board))                    #I push into history only the next_board
          push!(game.action_history, game.direction)
          push!(game.reward_history, game.reward)
          push!(game.done_history, game.lost)
end

# virtual step, does not change the game. It is only for evaluation purpose.
function virtual_step(game::SnakeGame, model::DQNModel)
    if game.lost
        push!(game.av_next_action_history, [CartesianIndex(0,0) for _ in 1:3])
        push!(game.next_is_suicidal_history, trues(3))
        return
    end

    assemble_state!(game)

    av_actions = available_actions(game)
    virtual_games = [deepcopy(game) for _ in 1:length(av_actions)]
    lost_vector = Bool[]

    for (gm, act) in zip(virtual_games, av_actions)
        step!(gm, act)
        push!(lost_vector, gm.lost)                                       #take care in training to the case in which all next actions are losing
    end

    push!(game.av_next_action_history, av_actions)
    push!(game.next_is_suicidal_history, lost_vector)
end


function assemble_state!(game::SnakeGame)    #fine
    last_frames = game.lost  ?  game.board_history[(end - game.n_frames):(end - 1)] : game.board_history[(end + 1 - game.n_frames):end]
    stacked = cat(last_frames...; dims=3)
    game.state = reshape(stacked, game.board_size, game.board_size, game.n_frames, 1)
end

function assemble_states_vector(game::SnakeGame)::Tuple{Vector{Array{Int,4}}, Vector{Array{Int,4}}}
    n_frames = game.n_frames
    len_game = length(game.board_history)
    all_states = [cat(game.board_history[i - n_frames + 1:i]...; dims=3) for i in n_frames:len_game]
    all_states_reshaped = [reshape(s, game.board_size, game.board_size, n_frames, 1) for s in all_states]
    states_vec = all_states_reshaped[1:end-game.n_frames]          #I am taking same number of states as the number of actions.(works 
    next_states_vec = all_states_reshaped[2:end-game.n_frames+1]
    return (states_vec, next_states_vec)
end

##################################################model methods################################################################

function epsilon_greedy(game::SnakeGame, model::Union{DQNModel, Chain}, epsilon::Float32)::CartesianIndex{2}

          av_actions = available_actions(game)
          push!(game.av_action_history, av_actions)               # For debugging
          
          assemble_state!(game)
          #@debug "assembled state" game.state
          
          if Float32(only(rand(1))) < epsilon         
              act = rand(av_actions)  
          else
          
             exp_rewards = typeof(model) == Chain ? temp_model(game.state) : model.q_net(game.state)
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

function play_episode(model::Union{DQNModel, Chain, Missing}, epsilon::Float32; actions_list::Union{Vector{CartesianIndex{2}},Missing}= missing)::Tuple{SnakeGame, Vector{Experience}, Float32}
    game = SnakeGame()
    episode_reward = 0.0f0
    
    if ismissing(actions_list)
    	while !game.lost
        	act = epsilon_greedy(game, model, epsilon)
        	step!(game, act)
        	virtual_step(game, model)   # suicide info for next state
        	episode_reward += game.reward
    	end
    else
    	for action in actions_list
    	        av_actions = available_actions(game)
    		#virtual_step(game, model)   # suicide info for next state
    		push!(game.av_next_action_history, [CartesianIndex(0,0) for _ in 1:3])   #I have to put something
                push!(game.next_is_suicidal_history, trues(3))                           #I have to put something
                push!(game.av_action_history, av_actions)
        	step!(game, action)
        	episode_reward += game.reward
        	if game.lost break end
    	end
    end

    # Attach last board n_frames - 1 times to fill final state
    push!(game.board_history, [deepcopy(game.board) for _ in 1:(game.n_frames - 1)]...)

    # Assemble states
    states_vec, next_states_vec = assemble_states_vector(game)

    # Validate experience vector lengths
    lengths = [
        length(states_vec),
        length(game.action_history),
        length(game.reward_history),
        length(next_states_vec),
        length(game.done_history),
        length(game.av_action_history),
        length(game.av_next_action_history),
        length(game.next_is_suicidal_history)
    ]

    if length(unique(lengths)) != 1
        error("Length mismatch in experience vectors: ", lengths)
    end

    exp_vec = [
        (s, a, r, ns, d, aa, naa, suicidal) 
        for (s, a, r, ns, d, aa, naa, suicidal) in zip(
            states_vec, 
            game.action_history, 
            game.reward_history, 
            next_states_vec, 
            game.done_history, 
            game.av_action_history,
            game.av_next_action_history,
            game.next_is_suicidal_history
        )
    ]
    
    return game, exp_vec, episode_reward
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
              return StatsBase.sample(rpb.buffer, length(rpb); replace = false)     
          else
              return StatsBase.sample(rpb.buffer, batch_size; replace = false)
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
         eps = ismissing(epsilon) ? 0.0f0 : epsilon
         while !isfull(rpb)
               _, exp_vec, _ = play_episode(model, eps)
               
               for exp in exp_vec
                    store!(rpb, exp)
               end
         end 
         @info "##############################  BUFFER FULL ######################################"
end 

function empty_buffer!(rpb::ReplayBuffer)
           rpb.buffer = Vector{Experience}(undef, 0)
           rpb.position = 1
end 

function save_buffer(rpb::ReplayBuffer, name::String)
    path = "/mnt/buffers/"
    if !isdir(path)
        mkpath(path)
    end

    file_path = joinpath(path, name * ".bson")
    BSON.@save file_path buffer=rpb
    println("Buffer saved to $file_path")
end

function load_buffer(name::String)
    path = "./buffers/"
    file_path = joinpath(path, name * ".bson")
    if !isfile(file_path)
        error("Buffer file not found at $file_path")
    end

    data = BSON.load(file_path)
    if !haskey(data, :buffer)
        error("Key `:buffer` not found in BSON file.")
    end
    println("Buffer loaded from $file_path")
    return data[:buffer]
end

###########################batch manipulation functions#################################################################################   
function stack_exp(batch::Vector{Experience})
    batch_size = length(batch)
    h, w = size(batch[1][1])  # board dimensions

    # Preallocate arrays
    states_array       = Array{Float32}(undef, h, w, 2, batch_size)             # 2 is n_frames
    next_states_array  = Array{Float32}(undef, h, w, 2, batch_size)
    actions_array      = Array{Int}(undef, batch_size)
    rewards_array      = Array{Float32}(undef, batch_size)
    done_array         = Array{Bool}(undef, batch_size)
    av_acts_array      = Array{Vector{CartesianIndex{2}}}(undef, batch_size)
    a_array            = Array{CartesianIndex{2}}(undef, batch_size)
    suicidal_mask      = Array{Bool}(undef, 3, batch_size)
    av_next_acts_array = Array{Vector{CartesianIndex{2}}}(undef, batch_size)

    for i in 1:batch_size
        s, a, r, s_next, done, av_acts, av_next_acts, suicidal_vec = batch[i]

        states_array[:, :, :, i]       .= Float32.(s)
        next_states_array[:, :, :, i]  .= Float32.(s_next)
        actions_array[i]  = findfirst(isequal(a), av_acts)
        rewards_array[i]  = Float32(r)
        done_array[i]     = done
        av_acts_array[i]  = av_acts
        a_array[i]        = a
        suicidal_mask[:, i] = suicidal_vec
        av_next_acts_array[i] = av_next_acts
    end

    return (
        states_array,
        actions_array,
        rewards_array,
        next_states_array,
        done_array,
        av_acts_array,
        a_array,
        suicidal_mask,
        av_next_acts_array,
    )
end


#############################################trainer methods#####################################################################

#method fill the buffer for the trainer, useful to debug
function fill_buffer!(tr::Trainer)
         @info "############################## FILLING THE BUFFER ###############################"
        
         while !isfull(tr.buffer)
                _, exp_vec, _ = play_episode(tr.model, tr.epsilon)
               
                for exp in exp_vec
                     store!(tr.buffer, exp)
                end            
         end 
         @info "##############################  BUFFER FULL ######################################"
end

function track_loss!(tr::Trainer, item)
          push!(tr.losses, item)
end

function save_trainer(tr::Trainer, name::String)
          path = "./trainers/"
          if !isdir(path) mkpath(path) end
          @save path * name * ".bson" tr
end

function load_trainer(name::String)::Trainer
          dir = "./trainers/"*name*".bson"
          @load dir tr
          return tr
end

function train!(tr::Trainer; trainer_name::String)
          
          #defining variables
          if tr.save log_hyperparameters(tr) end
      
          n_batches = tr.n_batches
          target_update_rate = tr.target_update_rate
          
          fill_buffer!(tr)
          opt_state = Flux.setup(tr.model.opt, tr.model.q_net)
          loss_val = 0.0 
          nb = 0
          
          @info "############################## START TRAINING ###############################"
          while nb <= n_batches
                 
                 _, exp_vec, episode_reward = play_episode(tr.model, tr.epsilon)
               
                 for exp in exp_vec
                      store!(tr.buffer, exp)
                 end         
                 
                 batch = sample(tr.buffer)
                 states, actions, rewards, next_states, dones, av_actions, a_array, suicidal_mask, av_next_acts_array  = stack_exp(batch)
                 #q_pred = tr.model.q_net(states)                                            # (n_actions, batch_size)
    		 #q_pred_selected = [q_pred[a, i] for (i, a) in enumerate(actions)]
    		 #q_pred_selected = reshape(q_pred_selected, :)                              # (batch_size,)
    		 
    		 q_next = tr.model.t_net(next_states)
    		 q_next[suicidal_mask] .= -100    
    		 max_next_q = dropdims(maximum(q_next, dims = 1), dims = 1)                 # (batch_size,)
		 q_target = @. rewards + 0.97 * max_next_q * (1 - dones)
                 
		 function loss_fun(z)
		           q_pred_selected = [z[a, i] for (i, a) in enumerate(actions)]
		           q_pred_selected = reshape(q_pred_selected, :)
		           loss_val = Flux.huber_loss(q_pred_selected, q_target; agg = mean)
		           return loss_val
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
		 
		 if nb % 5 == 0 
		     @printf "%d / %d -- episode_reward %.3f \n" nb n_batches episode_reward
		 end
		 
		 push!(tr.episode_rewards, episode_reward)		 
		 track_loss!(tr, loss_val)
		 tr.epsilon = max(tr.epsilon - tr.decay, tr.epsilon_end)                            #linear epsilon decay
		 nb += 1
          end 
                  
          @info "############################## END TRAINING ###############################"
          
          if tr.save 
              
              visualize(tr, trainer_name)
              save_buffer(tr.buffer, trainer_name)
              empty_buffer!(tr.buffer)                      #freeing up space
              save_trainer(tr, trainer_name) 
              @info "Trainer state saved to $trainer_name" 
          end  
end 

#########################################visualization and debugging##############################################################


#da fare: 1.funzione per giocare l'optimal policy  
#         2.istogramma del buffer 
#         3.potare l'explorazione in max Q-next                              fatto
#         4.funzione che plotta regret vs regret optimal policy
#         5.find a cheap way to save the buffer                              fatto
#         6.maybe exponential decay                                          no
#         7.use double frame as a state                                      fatto 
#         8.rewrite plot and play function                                   fatto
#         9.Aggiornare abstrcat e domanda di tesi                            Fatto
#         10.Funzione load_buffer                                            Fatto
#         11.Training                                                        (Domani)

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
        :board_size => game.board_size,
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


function plot_loss(tr::Trainer; size::Int = 100, name::Union{String, Missing} = missing)
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
    if !ismissing(name)
    
   	 if !isdir(path) mkpath(path) end
    	 savefig(pl, path * name * ".png")
    end
end

#plot average rewards
function plot_avg_rewards(tr::Trainer; size::Int=100, name::Union{String, Missing} = missing)
          
          pl = nothing
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
          
          if !ismissing(name)
          
          	if !isdir(path) mkpath(path) end
          	savefig(pl, path*name*".png")
          	return nothing
          end    
end

function play_best_game(tr::Union{Trainer, Missing}; name::Union{String, Missing}=missing, temp_model::Union{Chain, Nothing} = nothing)
          
          #if trainer is false it means that a temp_model has been passed
          plt = nothing
          
          model = ismissing(tr) ? missing : tr.model
          game, _, episode_reward = play_episode(model, 0.0f0)
         
          anim = @animate for board in game.board_history                                   #I want to exit the loop when game.lost
   
             #plot board
             plot_board(board, plt)
         end
         
         @printf "Final score %d \n" game.score
         path = "./trainer_gifs/"
         
         if !ismissing(name)
         
         	if !isdir(path) mkpath(path) end
         	gif(anim, path*name*".gif", fps=1)
         	
         end
         return nothing    
end

# Function to plot the game state
function plot_board(board::Matrix{Int}, plt::Union{Plots.Plot, Missing, Nothing} = missing)
    h, w = size(board)
    img = fill(ARGB32(1, 1, 1, 1), h, w)    # White background

    for i in 1:h, j in 1:w
        if board[i, j] == -1               # Walls
            img[i, j] = ARGB32(0, 0, 0, 1)  # Black
        elseif board[i, j] == 1            # Snake
            img[i, j] = ARGB32(0, 1, 0, 1)  # Green
        elseif board[i, j] == 2            # Food
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

function play_episode_with_animation(actions_list::Vector{CartesianIndex{2}};
                                     model::Union{DQNModel, Chain, Missing}=missing,
                                     epsilon::Float32=0.f0, 
                                     gif_name::Union{String, Nothing} = nothing,
                                     fps::Int = 1)

    game, exp_vec, episode_reward = play_episode(model, epsilon; actions_list = actions_list)

    anim = @animate for board in game.board_history
        plot_board(board)
    end every 1

    if gif_name !== nothing
        path = "./gifs/"
        if !isdir(path)
            mkpath(path)
        end
        gif(anim, path * gif_name * ".gif", fps = fps)
    end
    
    @printf "-- episode_reward %.3f \n" episode_reward
    return game, exp_vec, episode_reward, anim
end

#TODO:function to plot the buffer


function visualize(tr::Trainer, name::String)         #name is either save_name and load_name

          #rpb = load_buffer(name)
          #tr = load_trainer(name)
          plot_loss(tr; size = 100, name = name)
          plot_avg_rewards(tr; size = 100, name = name)
          play_best_game(tr; name=name, temp_model = nothing)
          plot_apple_histogram(tr.buffer; name= name)
end 

function count_apples_by_index(rpb::ReplayBuffer)
    
    game = SnakeGame()
    food_list = game.food_list
    count_per_index = zeros(Int, length(food_list))

    for exp in rpb.buffer
        state, _, reward, _, _, _, _, _ = exp

        if reward > 0
            pos = findfirst(x -> x == 2, state[:, :, end])  # food has value 2
            idx = findfirst(==(pos), food_list)155538407
            if idx !== nothing
                count_per_index[idx] += 1
            end
        end
    end

    return count_per_index
end

function plot_apple_histogram(rpb::ReplayBuffer; name::Union{String, Missing} = missing)
    counts = count_apples_by_index(rpb)
    pl = bar(1:length(counts),
        counts,
        xlabel = "Apple Index",
        ylabel = "Times Eaten",
        title = "# of apples in the buffer",
        legend = false,
        color = "red",
        xlims = (0, 21) 
        )
     
     path = "./buffer_histos/"   
     if !ismissing(name)
         
         	if !isdir(path) mkpath(path) end
         	savefig(pl, path*name*".png")
         	
     end
     return nothing    
end

