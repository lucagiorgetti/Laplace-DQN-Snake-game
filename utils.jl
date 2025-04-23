################################################## environment methods ###############################################

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
    return game.state[head] == -1 ||  count(==(head), game.snake) > 1 ||  game.prev_move + game.direction == CartesianIndex(0,0)   #true for a collision false otherwise, first condition check collision with the wall, second and third ones collisions with the snake itself.
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
        game.reward = 1 + game.discount * game.reward   #immediate reward = 1 for eating food
    else
        remove_tail!(game)  # Normal move (no food)
    end
end

#move wrapper, the snakes grow, if he hits the wall or himself he loses, if he does not run into food, I remove the tail 

function move_wrapper!(game::SnakeGame)
    grow_maybe!(game)  # Add new head
    
    if check_collision(game) 
        game.lost = true 
        game.reward = -1 + game.discount * game.reward   #immediate reward = -1 if he looses
    end

    update_state!(game)
    game.prev_move = game.direction
end

function get_step(game::SnakeGame, action::CartesianIndex{2})::Experience
         
          state = game.state
          game.direction = action
          move_wrapper!(game)
          reward = game.reward
          next_state = game.state
          if game.lost
             return (state, action, reward, next_state, true)
          else 
             return (state, action, reward, next_state, false)
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
         game = SnakeGame()
         while !isfull(rpb)
               # epsilon-greedy policy
               action = epsilon_greedy(game, model, 1.0f0)
               exp = get_step(game, action)
               store!(rpb, exp)
               if game.lost game = SnakeGame() end 
         end 
end 

function empty_buffer!(rpb::ReplayBuffer)
           rpb.buffer = Vector{Experience}(undef, 0)
end 

##################################################model methods################################################################

function epsilon_greedy(game::SnakeGame, model::DQNModel, epsilon::Float32)::CartesianIndex{2}

          actions = [CartesianIndex(-1,0), CartesianIndex(1,0), CartesianIndex(0,-1), CartesianIndex(0,1)]
          
          state = reshape(game.state, game.state_size, game.state_size, 1, 1)
          
          if Float32(only(rand(1))) < epsilon
              act = rand(actions)
          else
             exp_rewards = model.q_net(state)
             max_idx = argmax(exp_rewards)
             act = actions[max_idx]
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
          trainer = nothing
          @load dir trainer
          return trainer
end

function train!(tr::Trainer, trainer_name::String)
          
          #defining variables
          rpb = tr.buffer
          model = tr.model
          n_batches = tr.n_batches
          game = tr.game
          epsilon = tr.epsilon
          target_update_rate = tr.target_update_rate
          
          fill_buffer!(rpb, model)
          opt_state = Flux.setup(model.opt, model.q_net)
          nb = 0
          
          while nb <= n_batches
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
		 if tr.save 
		     track_loss!(tr, Flux.huber_loss(q_pred_selected, q_target))
		     end
		 epsilon = max(epsilon - tr.decay, tr.epsilon_end)                            #linear epsilon decay
		 nb += 1
          end 
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
          model = tr.model.q_net
          plt = plot_or_update!(game)
          sample_food!(game)
          
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
         
         path = "./gifs/"
         if !isdir(path) mkpath(path) end
         gif(anim, "./gifs/"*gif_name*".gif", fps=1)
         return nothing    
end
            
