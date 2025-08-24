# la_utils_cpu.jl
#########################################################################
# This is a CPU version of la_utils with persistent Laplace regime
# Modifications:
# 1. Plateau is checked every buffer.capacity batches
# 2. Once plateau is detected, Laplace regime stays active until K deviations
#    are collected and sampling is performed
# 3. After sampling, Laplace regime resets
#########################################################################

include("imports.jl")

mutable struct MeanStd 
    n::Int
    mean::Vector{Float64}
    m2::Vector{Float64}   # sum of squared deviations
end

# reset
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
    len_rewards = length(tr.episode_rewards)
    y = tr.episode_rewards[end - window : end] 
          
    if minimum(y) < -10
        return false
    end

    N = length(y)
    x = collect(len_rewards - window : len_rewards)
    X = hcat(ones(N), x)
    coeffs = X \ y                  # least squares
    slope = coeffs[2]
          
    if fig
        pl = plot(tr.episode_rewards, label = "Episode rewards", lw = 2, lc = :red)
        plot!(pl, x, @.(coeffs[2]*x + coeffs[2]), lw = 2, lc = :navy)
        xlabel!("episodes")
        ylabel!("rewards")
        display(pl)
    end
          
    println("slope is", slope)
    return -0.01 < slope < 0.01     # slope almost flat → apply Laplace
end

function compute_Gamma_diag(var::Vector{Float64})
    diag_elements = var
    if minimum(diag_elements) < 0
        @warn "Gamma_diag has negative element, value = $(minimum(diag_elements))"
        diag_elements = abs.(diag_elements)
    end
    return Diagonal(diag_elements)
end

function sample_model(mean::Array{Float64}, var::Array{Float64}, D::Matrix{Float64}, restructure)
    Gamma_diag = compute_Gamma_diag(var)
    d = length(mean) # total size of the model
    K = size(D, 2) == 6 ? 6 : @error "Size of D is not 6, but $(size(D,2))"
    nd = MvNormal(fill(.0, d), I)   
    nK = MvNormal(fill(.0, K), I)
          
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

function resume_training!(; n_batches::Int=500_000, trainer_name::String, la_trainer_name::String)
    tr = load_trainer(trainer_name)
    tr.buffer.batch_size = 1024
          
    if tr.save
        log_hyperparameters(tr)
    end
      
    target_update_rate = tr.target_update_rate
    theta_init, re = Flux.destructure(tr.model.q_net)
    param_count = length(theta_init)
          
    fill_buffer!(tr)
          
    K = 6                                          
    thin = 1000
    position = 1
    laplace = false                                  
    deviation_matrix = zeros(Float64, (param_count, K))
    o = MeanStd(param_count)
          
    nb = 1
    @info "############################## START TRAINING ###############################"
    while nb <= n_batches
        # check for plateau every buffer.capacity steps
        if nb % tr.buffer.capacity == 0 && !laplace
            laplace = check_plateau(tr; window = 2000)
            if laplace
                @info "Plateau detected at batch $nb — entering Laplace regime"
            end
        end

        # collect deviations every 'thin' steps if Laplace regime is active
        if laplace && (nb % thin == 0) && (position <= K)
            theta, _ = Flux.destructure(tr.model.q_net)
            deviation_matrix[:, position] = Float64.(theta)
            position += 1
        end

        # once K deviations are collected, run Laplace sampling
        if laplace && (position == K + 1)
            avg = mean(o)
            var = var(o)
            deviation_matrix .-= avg[:, ones(K)]
            laplace_sampling!(tr, avg, var, deviation_matrix, re=re)
            deviation_matrix .= 0.0
            o = MeanStd(param_count)
            position = 1
            laplace = false
            @info "Laplace sampling done at batch $nb — leaving Laplace regime"
        end

        # === normal training step ===
        _, exp_vec, episode_reward = play_episode(tr.model, tr.epsilon)
        for exp in exp_vec
            store!(tr.buffer, exp)
        end
                 
        batch = sample(tr.buffer)
        states, actions, rewards, next_states, dones, av_actions, a_array, suicidal_mask, av_next_acts_array  = stack_exp(batch)
     
        q_next = tr.model.t_net(next_states)
        q_next[suicidal_mask] .= -100    
        max_next_q = dropdims(maximum(q_next, dims = 1), dims = 1)
        q_target = @. rewards + 0.97 * max_next_q * (1 - dones)

        q_pred = tr.model.q_net(states)
        batch_size = length(actions)
        num_actions = size(q_pred, 1)
        linear_inds = actions .+ (0:batch_size-1) .* num_actions
        q_pred_selected = q_pred[linear_inds]
        q_pred_selected = reshape(q_pred_selected, :)
		 
        function loss_fun(z)
            batch_size = length(actions)
            num_actions = size(z, 1)
            linear_inds = actions .+ (0:batch_size-1) .* num_actions
            q_pred_selected = z[linear_inds]
            q_pred_selected = reshape(q_pred_selected, :)
            return Flux.huber_loss(q_pred_selected, q_target)
        end
		 
        grads = Flux.gradient(tr.model.q_net) do m
            q_pred = m(states)
            loss_fun(q_pred)
        end

        Flux.update!(Flux.setup(tr.model.opt, tr.model.q_net), tr.model.q_net, only(grads))
        if isnothing(only(grads))
            @warn "Network has not been updated"
        end
                 
        if nb % target_update_rate == 0 
            update_target_net!(tr.model) 
            @info "Batch $nb | Target network updated." 
        end
		
        if nb % 5 == 0 
            @printf "%d / %d -- episode_reward %.3f \n" nb n_batches episode_reward
        end
		 
        push!(tr.episode_rewards, episode_reward)	
        loss_val = Flux.huber_loss(q_pred_selected, q_target) |> float	 
        track_loss!(tr, loss_val)
        tr.epsilon = max(tr.epsilon - tr.decay, tr.epsilon_end)                            
		
        nb += 1
    end 
                  
    @info "############################## END TRAINING ###############################"


