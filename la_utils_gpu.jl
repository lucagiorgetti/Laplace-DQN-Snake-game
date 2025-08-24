#la_utils_gpu.jl
#########################################################################
#This is a version of la_utils.jl that runs on a gpu.
#Main changes:
#1. Move model and data on a gpu
#2. Remove scalar indexing
############################################################################ 

include("imports.jl")
using CUDA

mutable struct MeanStd 
    n::Int
    mean::Vector{Float64}
    m2::Vector{Float64}   # sum of squared deviations
end

#this basically a reset
function MeanStd(d::Int)
    MeanStd(0, zeros(d), zeros(d))
end

# one observation (a column vector)
function fit!(o::MeanStd, x::AbstractVector)
    o.n += 1
    Δ = x .- o.mean
    o.mean .+= Δ ./ o.n
    o.m2 .+= Δ .* (x .- o.mean)
    return o
end

mean(o::MeanStd) = o.mean
var(o::MeanStd)  = o.m2 ./ max(o.n - 1, 1)
std(o::MeanStd)  = sqrt.(var(o))

function save_buffer(rpb::ReplayBuffer, name::String)
    path = "./buffers/"
    if !isdir(path)
        mkpath(path)
    end

    file_path = joinpath(path, name * ".bson")
    BSON.@save file_path buffer=rpb
    println("Buffer saved to $file_path")
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

function compute_Gamma_diag(var::Vector{Float64})
         diag_elements = var
         if minimum(diag_elements) < 0
            @warn "Gamma_diag has negative element, whose value is $(minimum(diag_elements))"
            diag_elements = abs.(diag_elements)
         end
         return Diagonal(diag_elements)
end

function sample_model(mean::Array{Float64}, var::Array{Float64}, D::Matrix{Float64}, restructure)
          
          Gamma_diag = compute_Gamma_diag(var)
          d = length(mean)                                                              #total size of the model
          K = size(D)[2] == 6 ? 6 : @error "Size of D is not 6, but $size(D)[2]" 
          nd = MvNormal(fill(.0,d), I)   
          nK = MvNormal(fill(.0,K), I)
          
          z1 = rand(nd)
          z2 = rand(nK) 
          
          w = mean + 1/sqrt(2) * sqrt.(Gamma_diag) * z1 + 1/sqrt(2*(K - 1)) * D * z2
          return restructure(w)    
end 

function laplace_sampling!(tr::Trainer, mean::Vector{Float64}, var::Vector{Float64}, D::AbstractMatrix; n_models::Int=5000, epsilon::Float32=0.0f0, re)
          
           @info "Starting sampling LA models" 
          
           _, _, tr_reward = play_episode(missing, 0.f0)
           n_better_models = 0
           
           for n in 1:n_models
                 
                 model = sample_model(mean, var, D, re)
                 _, exp, episode_reward = play_episode(model, 0.0f0)
                 
                 if episode_reward >= tr_reward
                     n_better_models += 1
                     store!(tr.rpb, exp)
                 end
           end
   
          @info "end sampling LA models, n_better_models:" n_better_models
end

function resume_training!(;n_batches::Int=500_000, trainer_name::String, la_trainer_name::String) #trainer_path for loading, la_trainer_name for saving
          
          tr = load_trainer(trainer_name)
          tr.buffer.batch_size = 1024
          
          #defining variables
          if tr.save log_hyperparameters(tr) end
      
          target_update_rate = tr.target_update_rate
          theta_init, re = Flux.destructure(tr.model.q_net)
          param_count = length(theta_init)
          
          #filling the buffer
          fill_buffer!(tr)
          
          #defining the deviation matrix and the Laplace boolean variable
         
          K = 6                                          
          thin = 1000
          position = 1
          laplace = false                                  
          deviation_matrix = zeros(Float64, (param_count, K))
          o = MeanStd(param_count)
          
          # move model + optimizer to GPU
    	  tr.model.q_net = gpu(tr.model.q_net)
          tr.model.t_net = gpu(tr.model.t_net)
    	  opt_state = Flux.setup(tr.model.opt, tr.model.q_net)
          
          nb = 1
          
          @info "############################## START TRAINING ###############################"
          while nb <= n_batches
                 
                #deciding whether I am in Laplace regime (I check every 50_000 mini-batches)
                laplace = nb % tr.buffer.capacity == 0 ? check_plateau(tr; window = 2000) : false
                 
                #If Laplace regime and D matrix is full, I add some Laplace samples to the buffer
                if position == K + 1
                 
                     avg = mean(o)
                     var = var(o)
                     
                     deviation_matrix = deviation_matrix .- avg
                     laplace_sampling!(cpu(tr), avg, var, deviation_matrix, re=re)  
                     
                     #resetting vars
                     deviation_matrix = zeros(Float64, (param_count, K))
                     o = MeanStd(param_count)
                     position = 1
                     laplace = false
                        
                end
                       
                _, exp_vec, episode_reward = play_episode(cpu(tr.model), tr.epsilon)
               
                for exp in exp_vec
                    store!(tr.buffer, exp)
                end 
                 
                batch = sample(tr.buffer)
                states, actions, rewards, next_states, dones, av_actions, a_array, suicidal_mask, av_next_acts_array  = stack_exp(batch)
                 
                # === Move minibatch to GPU ===
        	    states = cu(states)
        	    next_states = cu(next_states)
        	    rewards = cu(rewards)
        	    dones = cu(dones)
        	    suicidal_mask = cu(suicidal_mask)
     
    		    q_next = tr.model.t_net(next_states)
    		    q_next[suicidal_mask] .= -100    
    		    max_next_q = dropdims(maximum(q_next, dims = 1), dims = 1)                 # (batch_size,)
		        q_target = @. rewards + 0.97 * max_next_q * (1 - dones)

                #for tracking the loss
                q_pred = tr.model.q_net(states)
                batch_size = length(actions)
                num_actions = size(q_pred, 1)
                linear_inds = actions .+ (0:batch_size-1) .* num_actions
                q_pred_selected = q[linear_inds]
                q_pred_selected = reshape(q_pred_selected, :)
		 
		        function loss_fun(z)
		            batch_size = length(actions)
            		num_actions = size(z, 1)
            
            		# compute linear indices for (actions, 1:batch_size)
                    linear_inds = actions .+ (0:batch_size-1) .* num_actions
            
                    # gather all at once on GPU
                    q_pred_selected = z[linear_inds]
		            q_pred_selected = reshape(q_pred_selected, :)
		            return Flux.huber_loss(q_pred_selected, q_target)
		        end
		 
		        #doing the update
		        grads = Flux.gradient(tr.model.q_net) do m
		            q_pred = m(states)
                    loss_fun(q_pred)
                end

                Flux.update!(opt_state, tr.model.q_net, only(grads))
                if isnothing(only(grads)) @warn "Network has not been updated" end
                 
                #if Laplace Regime update momenta and add column to D, and apply thinning
                if laplace && (nb % thin == 0) && (position <= K)
                    theta, _ = Flux.destructure(cpu(tr.model.q_net))
                    theta = Float64.(theta)
                    dD = theta 
                    
                    deviation_matrix[:, position] = dD             
                    position += 1              
                 end
		 
		        if nb % target_update_rate == 0 
		            update_target_net!(tr.model) 
		            @info "Batch $nb | Target network updated." 
		        end
		
		 
		        if nb % 5 == 0 
		            @printf "%d / %d -- episode_reward %.3f \n" nb n_batches episode_reward
		        end
		 
		        push!(tr.episode_rewards, episode_reward)	
                loss_val = Flux.huber_loss(q_pred_selected, q_target) |> cpu |> float	 
		        track_loss!(tr, loss_val)
		        tr.epsilon = max(tr.epsilon - tr.decay, tr.epsilon_end)                            
		
		        nb += 1
            end 
                  
          @info "############################## END TRAINING ###############################"
          
          save_buffer(tr.buffer, la_trainer_name)
          empty_buffer!(tr.buffer)                      #freeing up space
          save_trainer(cpu(tr), la_trainer_name) 
          @info "Trainer state saved to $trainer_name"   

end

# === Run ===
resume_training!(n_batches=500_000, trainer_name="very_long_double_training3", la_trainer_name="very_long_la_double_training3")

