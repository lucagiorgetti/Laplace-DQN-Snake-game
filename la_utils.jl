#la_utils.jl
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
function welford_update(count::Int, aggregate::Tuple{Vector{Float64}, Vector{Float64}}, new_value::Vector{Float64})
          
          mean, m2 = aggregate
          count += 1
          delta = @.(new_value - mean)
          mean += delta / count
          delta2 = @.(new_value - mean)
          m2 += @.(delta * delta2)
          return count, mean, m2
end

function welford_finalize(count::Int, aggregate::Tuple{Vector{Float64}, Vector{Float64}})
          
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
                 end  
                 
                 #If Laplace regime and D matrix is already full I can start sample actions from posterior, otherwise epsilon-greedy
                 if laplace&&laplace_counter >= treshold
                     
                     if (laplace_counter - treshold) % 500 == 0 
                         @info "sampling a model for LA" 
                         theta_SWA, var_SWA = welford_finalize(laplace_counter, (theta_SWA, m2))
                         temp_model = sample_model_with_memory_mapping(theta_SWA, var_SWA, re; D = deviation_matrix)
                         
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
		  	
		 if !laplace
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
         fill_buffer!(bf, md; epsilon = 0.0f0)
         return bf
end 

#################################################################################################################
#No temp model works but the very first. My interpretation is that:
#1. We are not around a minimum and the model keeps evolving
#2. The model is jumping from minimum to minimum and is crucial to not accumulate too much variance in D
#Let's instead try to freeze the training during Laplace (equivalently, sample all the models together, but this requires too much memory.)
#Keeping a bit of stochasticity should help to avoid loops.
#Maybe a level up is to actually to reason in terms of games and not samples of experience and to abort games when last too long.
##################################################################################################################

#function to compute the number of iteration per la_model and the number of models. Consider that I want to fill only 60 percent of the buffer with Laplace samples
function pre_la(tr::Trainer)
         
         capacity = tr.buffer.capacity
         size_to_fill = floor(0.6 * capacity) 
         
         #I want roughly 10 games per model, let's say 1000 samples per model
         n_samples_per_model = 1000
         
         n_models = div(size_too_fill, n_samples_per_model)
         return n_models, n_samples_per_model)
end

#function to fill the buffer with Laplace samples

function add_samples_to_buffer!(rb::ReplayBuffer, md::Chain;n_samples::Int, epsilon::Float32=0.1f0)
          game = SnakeGame()
         
          for n in 1:n_samples
          
               # epsilon-greedy policy
               action = epsilon_greedy(game, missing, epsilon; temp_model = md)
               exp = get_step(game, action)
               store!(rb, exp)
               if game.lost game = SnakeGame() end               
         end 
end

function laplace_sampling!(tr::Trainer; mean::Vector{Float64}, m2::Vector{Float64}, D::AbstractMatrix, n_models::Int, n_samples::Int, epsilon::Float32=0.1f0, count::Int, re)
          
           @info "Starting samplin LA models" 
           path = "./temp_models/"
           if !isdir(path) mkpath(path) end
           
           theta_SWA, var_SWA = welford_finalize(counter, (mean, m2))
           for n in 1:n_models
         
               temp_model = sample_model_with_memory_mapping(theta_SWA, var_SWA, re; D = D)
               add_samples_to_buffer!(tr.buffer, temp_model;n_samples = n_samples, epsilon=epsilon)     
                    
               #saving temp models for further analysis (in particular I need to inspect the buffer later)
               @save path * "_temp_model_$n" * ".bson" temp_model
            end 
          @info "end sampling LA models"
end

function reset_laplace_vars(file::IOStream, file_name::String)

         if file !== nothing && isopen(deviation_file)
             close(deviation_file)
         end
         if isfile(file_name)
             rm(file_name; force = true)
         end
         
         return 0, false   #new value for counter and Laplace
end

function resume_training_mod!(;n_batches::Int=100000, trainer_path::String, la_trainer_name::String) #trainer_path for loading, la_trainer_name for saving
          
          n_models, n_samples = pre_la(tr)
          
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
         
          capacity = 1000     #number of columns of the deviation mat
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
          laplace = false
          
          nb = 0
          
          @info "############################## START TRAINING ###############################"
          while nb <= n_batches
                 
                 #deciding whether I am in Laplace regime
                 laplace = nb % 2*tr.buffer.capacity == 0 ? check_plateau(tr; window = 2000) : false
                 
                 #if Laplace regime starting to initialize first and second moments of the weights
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
                 end  
                 
                 #If Laplace regime and D matrix is already full I can start sample actions from posterior, otherwise epsilon-greedy
                 if laplace&&laplace_counter >= treshold
                  
                    laplace_sampling!(tr; mean=theta_SWA, m2=m2, D=deviation_matrix, n_models=n_models, n_samples=n_samples, epsilon=0.1f0, count=laplace_counter, re=re)
                    
                      laplace_counter,laplace = reset_laplace_vars(deviation_file, "deviation_matrix.bin")    
                        
                 else
                       
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
		 
		 if laplace
		     laplace_counter += 1	
		 else
		      tr.epsilon = max(tr.epsilon - tr.decay, tr.epsilon_end)                            #linear epsilon decay
		 end
		
		 nb += 1
          end 
                  
          @info "############################## END TRAINING ###############################"
          
          if tr.save 
              
              plot_and_play(tr; save_name = la_trainer_name)
              empty_buffer!(tr.buffer)                      #freeing up space
              save_trainer(tr, la_trainer_name) 
              @info "Trainer state saved to $la_trainer_name" 
          end

end

