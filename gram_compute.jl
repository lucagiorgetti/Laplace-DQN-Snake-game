####################################################################
# This script is meant to find the optimal K parameter of SWAG
####################################################################

include("imports.jl")

name = "very_long_double_training3"
tr = load_trainer(name)
bf = load_buffer(name)

tr.buffer = bf

# Taking bigger batch size in order to have a better representation of the loss, it was 10_000
tr.buffer.batch_size = 1000



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

function compute_gram(tr::Trainer, name::String)

    param_count = length(Flux.destructure(tr.model.q_net)[1])

    # Deviation matrix, let's start with a quite big one 
    # (for Marchenko-Pastur should go to infinity), it was 1_000
    T = 1_000
    D = zeros(Float64, T, param_count)
    thin = 1_000
    n_batches = 1_000_000
    
    # Initializing variables
    theta_SWA = zeros(param_count)  # Initialization for Welford's algorithm
    m2 = zeros(param_count)
    laplace_counter = 0
    nb = 1
    store_index = 1
    opt_state = Flux.setup(tr.model.opt, tr.model.q_net)

    # I need an almost stable buffer; I do not add other games to it
    # while I am building the Gram matrix
    while (nb <= n_batches) && (store_index <= T)

        batch = sample(tr.buffer)
        states, actions, rewards, next_states, dones, av_actions, a_array, suicidal_mask, av_next_acts_array =
            stack_exp(batch)

        q_next = tr.model.t_net(next_states)
        q_next[suicidal_mask] .= -100
        max_next_q = dropdims(maximum(q_next, dims = 1), dims = 1)  # (batch_size,)
        q_target = @. rewards + 0.97 * max_next_q * (1 - dones)

        function loss_fun(z)
            q_pred_selected = [z[a, i] for (i, a) in enumerate(actions)]
            q_pred_selected = reshape(q_pred_selected, :)
            loss_val = Flux.huber_loss(q_pred_selected, q_target; agg = mean)
            return loss_val
        end

        # Doing the update
        grads = Flux.gradient(tr.model.q_net) do m
            q_pred = m(states)
            loss_fun(q_pred)
        end

        Flux.update!(opt_state, tr.model.q_net, only(grads))

        if isnothing(only(grads))
            @warn "Network has not been updated"
        end

        if nb % tr.target_update_rate == 0
            update_target_net!(tr.model)
            @info "Batch $nb | Target network updated."
        end

        # TODO: Fill the deviation matrix using mean_i and then, at the end,
        # divide by the best estimate of the variance.
        
        theta, _ = Flux.destructure(tr.model.q_net)
        theta = Float64.(theta)
        laplace_counter, theta_SWA, m2 = welford_update(laplace_counter, (theta_SWA, m2), theta)
            
        if (nb % thin)  == 0
            
            dD = theta - theta_SWA
            D[store_index, :] = dD
            
            if laplace_counter % 5 == 0
                @printf "store_index: %d\n" store_index
            end
            store_index += 1
        end
        
        nb += 1
    end

    # Welford finalize and divide each entry by sigma^2
    theta_SWA, var_SWA = welford_finalize(laplace_counter, (theta_SWA, m2))
    
    epsilon = 1e-8
    D = D ./ sqrt.(var_SWA .+ epsilon)   # Normalize each parameter's deviations
    Gr = (D * D') / param_count
  
    if any(Gr .== 0)
        throw("Something wrong filling D")
    end
    
    # --- Independence check ---
    offdiag = Gr .- Diagonal(diag(Gr))
    mean_abs_offdiag = mean(abs.(offdiag))
    max_abs_offdiag  = maximum(abs.(offdiag))
    @info "Row independence: mean|offdiag|=$(round(mean_abs_offdiag, digits=4)) " *
      "max|offdiag|=$(round(max_abs_offdiag, digits=4))"
    # --- End of independence check ---

    path = "./gram_matrix/"
    if !isdir(path)
        mkpath(path)
    end

    file_path = joinpath(path, name * ".bson")
    BSON.@save file_path gram_matrix = Gr
    println("Gram matrix saved to $file_path")

    return nothing
end

compute_gram(tr, name)

