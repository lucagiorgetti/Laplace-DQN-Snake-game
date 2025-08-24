include("imports.jl")
using CUDA

function compute_gram_gpu(tr::Trainer, name::String; 
                          n_batches::Int = 1_000_000, 
                          thin::Int = 1_000,
                          γ::Float64 = 0.97)

    # number of snapshots
    n_snaps = div(n_batches, thin)

    # number of parameters
    param_count = length(Flux.destructure(tr.model.q_net)[1])

    # allocate snapshot matrix in CPU RAM (no mmap)
    Q_weights = Matrix{Float64}(undef, param_count, n_snaps)

    # move model + optimizer to GPU
    tr.model.q_net = gpu(tr.model.q_net)
    tr.model.t_net = gpu(tr.model.t_net)
    opt_state = Flux.setup(tr.model.opt, tr.model.q_net)

    nb = 1
    store_index = 1

    while nb <= n_batches
        # === Sample minibatch (CPU) ===
        batch = sample(tr.buffer)
        states, actions, rewards, next_states, dones,
        av_actions, a_array, suicidal_mask, av_next_acts_array = stack_exp(batch)

        # === Move minibatch to GPU ===
        states = cu(states)
        next_states = cu(next_states)
        rewards = cu(rewards)
        dones = cu(dones)
        suicidal_mask = cu(suicidal_mask)

        # === Target computation on GPU ===
        q_next = tr.model.t_net(next_states)
        q_next[suicidal_mask] .= -100.0
        max_next_q = dropdims(maximum(q_next, dims = 1), dims = 1)
        q_target = @. rewards + γ * max_next_q * (1 - dones)

        # === Loss function ===
        function loss_fun(z)
            batch_size = length(actions)
            num_actions = size(z, 1)
            
            # compute linear indices for (actions, 1:batch_size)
            linear_inds = actions .+ (0:batch_size-1) .* num_actions
            
            # gather all at once on GPU
            q_pred_selected = z[linear_inds]
            
            q_pred_selected = reshape(q_pred_selected, :)
            Flux.huber_loss(q_pred_selected, q_target; agg = mean)
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

        # === Store snapshot ===
        if nb % thin == 0
            theta, _ = Flux.destructure(cpu(tr.model.q_net))  # ensure CPU Float64
            Q_weights[:, store_index] = Float64.(theta)
            store_index += 1
        end

        nb += 1
    end

    # === Compute covariance matrix ===
    o = CovMatrix(n_snaps)
    fit!(o, Q_weights |> eachrow)

    # === Save results ===
    path = "./gram_stats/"
    mkpath(path)
    file_path = joinpath(path, name * ".bson")
    BSON.@save file_path gram_matrix = o
    println("Gram stats saved to $file_path")

    return nothing
end

# === Run ===
name = "very_long_double_training3"
tr = load_trainer(name)
bf = load_buffer(name)
# batch size → large for GPU throughput
tr.buffer.batch_size = 1024
tr.buffer = bf

compute_gram_gpu(tr, name)
