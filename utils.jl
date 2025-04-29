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
        return  # No space left
    end

    # Pick a random empty spot
    food_pos = rand(game.rng, empty_positions)

    # Update game state
    game.state[food_pos] = 2
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
        sample_food!(game)
        game.score += 1
        game.reward = 1.0f0                     #immediate reward = 1 for eating food
        @info "food in ($(new_head[1]),$(new_head[2])) eaten!"
    else
        remove_tail!(game)  # Normal move (no food)
        game.reward = -0.001f0                #small penalty for surviving without eating anything
    end
end

#move wrapper, the snakes grow, if he hits the wall or himself he loses, if he does not run into food, I remove the tail 

function move_wrapper!(game::SnakeGame)
    grow_maybe!(game)  # Add new head
    
    if check_collision(game) 
        game.lost = true 
        game.reward = -1.0f0                     #immediate reward = -1 if he looses
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

function fill_buffer!(rpb::ReplayBuffer, model::DQNModel)
         println("############################## FILLING THE BUFFER ###############################")
         game = SnakeGame()
         while !isfull(rpb)
               # epsilon-greedy policy
               action = epsilon_greedy(game, model, 1.0f0)
               exp = get_step(game, action)
               store!(rpb, exp)
               if game.lost game = SnakeGame() end 
         end 
         println("##############################  BUFFER FULL ######################################")
end 

function empty_buffer!(rpb::ReplayBuffer)
           rpb.buffer = Vector{Experience}(undef, 0)
           rpb.position = 1
end 

##################################################model methods################################################################

function epsilon_greedy(game::SnakeGame, model::DQNModel, epsilon::Float32; debug::Bool=false)::CartesianIndex{2}

          av_actions = available_actions(game)
          
          state = reshape(game.state, game.state_size, game.state_size, 1, 1)
          
          if Float32(only(rand(1))) < epsilon
              act = rand(av_actions)
          else
             exp_rewards = model.q_net(state)
             max_idx = argmax(exp_rewards)
             act = av_actions[max_idx]
             if debug
              println("--------------------EPSILON GREEDY LOGGIN-----------------------------")
              println("--------------------------------------------------------")
              println("q_values: ", exp_rewards)
              println("argmax: ", max_idx)
              println("--------------------------------------------------------")
              println("--------------------END EPSILON GREEDY LOGGIN-----------------------------")
          end
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

function load_model(dir::String)::DQNModel
    model = nothing
    @load dir model
    return model
end

###########################batch manipulation functions#################################################################################   
"""
function action_to_index(a::CartesianIndex{2})
    return ACTIONS[a]
end
"""

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
        s, a, r, s_next, done, av_acts = batch[i]

        states_array[:, :, 1, i]      .= Float32.(s)
        next_states_array[:, :, 1, i] .= Float32.(s_next)
        actions_array[i]  = findfirst(isequal(a), av_acts)
        rewards_array[i]  = Float32(r)
        done_array[i]     = done
    end

    return (states_array, actions_array, rewards_array, next_states_array, done_array)
end

#############################################trainer methods#####################################################################
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

function train!(tr::Trainer, trainer_name::String)
          
          #defining variables
      
          n_batches = tr.n_batches
          game = tr.game
          target_update_rate = tr.target_update_rate
          
          fill_buffer!(tr.buffer, tr.model)
          opt_state = Flux.setup(tr.model.opt, tr.model.q_net)
          nb = 0
          
          println("############################## START TRAINING ###############################")
          while nb <= n_batches
                 action = epsilon_greedy(game, tr.model, tr.epsilon)
                 experience = get_step(game, action)
                 store!(tr.buffer, experience)
                 
                 batch = sample(tr.buffer)
                 states, actions, rewards, next_states, dones = stack_exp(batch)
                 q_pred = tr.model.q_net(states)                                            # (n_actions, batch_size)
    		 q_pred_selected = [q_pred[a, i] for (i, a) in enumerate(actions)]
    		 q_pred_selected = reshape(q_pred_selected, :)                              # (batch_size,)
    		 q_next = tr.model.t_net(next_states)
    		 max_next_q = dropdims(maximum(q_next, dims = 1), dims = 1)                 # (batch_size,)
		 q_target = @. rewards + game.discount * max_next_q * (1 - dones)
		 
		 function loss_fun(z)
		           q_pred_selected = [z[a, i] for (i, a) in enumerate(actions)]
		           q_pred_selected = reshape(q_pred_selected, :) 
		           return Flux.huber_loss(q_pred_selected, q_target)
		 end
		 
		 #doing the update
		 grads = Flux.gradient(tr.model.q_net) do m
		       q_pred = m(states)
                       loss_fun(q_pred)
                 end

                 Flux.update!(opt_state, tr.model.q_net, grads[1])
                 if isnothing(grads[1]) @warn "Network has not been updated" end
		 
		 if nb % target_update_rate == 0 update_target_net!(tr.model) end
		 if game.lost game = SnakeGame() end
		 if nb % 5 == 0 @printf "%d / %d -- loss %.3f \n" nb n_batches Flux.huber_loss(q_pred_selected, q_target) end
		 if tr.save 
		     track_loss!(tr, Flux.huber_loss(q_pred_selected, q_target))
		     end
		 tr.epsilon = max(tr.epsilon - tr.decay, tr.epsilon_end)                            #linear epsilon decay
		 nb += 1
          end         
          println("############################## END TRAINING ###############################")
          if tr.save save_trainer(tr, trainer_name) end  
end 

############################################################visualization functions##############################################

function plot_loss(tr::Trainer; mv_avg::Bool = true, size::Int = 10, save_name::String)
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
    if mv_avg && length(y) ≥ size
        z = [mean(y[i-size+1:i]) for i in size:length(y)]
        x_avg = x[size:end]
        plot!(x_avg, z, label = "Moving Avg", lw = 2, lc = :red)
    end
    
    if tr.save
        path = "./plots/"
        if !isdir(path) mkpath(path) end
        savefig(pl, path * save_name * ".png")
    end
end

function play_game(tr::Trainer, gif_name::String)
          plt = nothing
          game = SnakeGame()
          model = tr.model
          plt = plot_or_update!(game)
          #sample_food!(game)                                               #I am placing the first directly from inside the struct
          
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
                 @printf "Collision!! Final score %d, reward %.3f \n" game.score game.reward
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
