#compute_D.jl
###################################################################
#this function computes the deviation_matrix
####################################################################
include("imports.jl")

BLAS.set_num_threads(2)

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

function compute_D(trainer_name::String, save_name::String)

    save_path = "./D_matrices/"
    if !isdir(save_path) 
        mkpath(save_path) 
    end
    
    tr = load_trainer(trainer_name)
    target_update_rate = tr.target_update_rate
    n_batches = tr.n_batches
    theta_init, re = Flux.destructure(tr.model.q_net)
    param_count = length(theta_init)

    fill_buffer!(tr)
    opt_state = Flux.setup(tr.model.opt, tr.model.q_net)
    
    thin = 10
    K = 1000
    position = 1
    laplace = false
    deviation_matrix = zeros(Float64, (param_count, K))
    o = MeanStd(param_count)

    nb = 1
    @info "############################## START TRAINING ###############################"
    while nb <= n_batches
        
        #burn_in
        if nb == 50_000
            laplace = true
            @info "Starting to fill D"
        end

        # collect deviations every 'thin' steps if Laplace regime is active
        if laplace && (position <= K) && (nb % thin == 0)
            theta, _ = Flux.destructure(tr.model.q_net)
            deviation_matrix[:, position] = Float64.(theta)
            position += 1
        end

        # once K deviations are collected, run Laplace sampling
        if laplace && (position == K + 1)
            
            for c in eachcol(deviation_matrix)
               fit!(o, c)
            end
            
            avg = mean(o)
            deviation_matrix .-= avg
            
            #saving D
            BSON.@save save_path * save_name * ".bson" deviation_matrix = deviation_matrix
            @info "Deviation matrix saved"
            return nothing
        end

        # === normal training step ===
        _, exp_vec, episode_reward = play_episode(tr.model, tr.epsilon)
        for exp in exp_vec
            store!(tr.buffer, exp)
        end

        batch = sample(tr.buffer)
        states, actions, rewards, next_states, dones, av_actions, a_array, suicidal_mask, av_next_acts_array = stack_exp(batch)

        q_next = tr.model.t_net(next_states)
        q_next[suicidal_mask] .= -100
        max_next_q = dropdims(maximum(q_next, dims=1), dims=1)
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

        Flux.update!(opt_state, tr.model.q_net, only(grads))
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

        tr.epsilon = max(tr.epsilon - tr.decay, tr.epsilon_end)

        nb += 1
    end
end

compute_D("very_long_double_training3", "D_very_long_double_training3")
