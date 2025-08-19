#gram_compute2.jl
##############################################################################
#This script computes the Gram matrix of the weights of the Q-net using OnlineStats
##################################################################################

include("imports.jl")

name = "very_long_double_training3"

function load_buffer(name::String)
    path = "/mnt/buffers/"
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

function save_theta!(tr::Trainer, name::String; dir::AbstractString="/mnt/Q_weights", overwrite::Bool=true)
    mkpath(dir)  # make sure directory exists
    file_path = joinpath(dir, string(name, ".csv"))

    # If a directory with that name exists, refuse — user likely made the earlier mistake.
    if isdir(file_path)
        error("Refusing to write: '$file_path' exists and is a directory. Remove it or choose a different name.")
    end

    if overwrite && isfile(file_path)
        rm(file_path)
    end

    # get parameters
    theta, _ = Flux.destructure(tr.model.q_net)
    theta = Float64.(theta)  # ensure Float64

    # format with high precision to avoid precision loss when reloading
    line = join([@sprintf("%.17g", x) for x in theta], ",")

    open(file_path, "a") do io
        println(io, line)    # writes the CSV row + newline, file auto-closed
    end

    return file_path
end

function compute_gram(tr::Trainer, name::String)

    n_batches = 200_000
    
    # Initializing variables
    nb = 1
    n_snaps = 0
    opt_state = Flux.setup(tr.model.opt, tr.model.q_net)
    thin = 100

    while nb <= n_batches
    
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

        if nb % thin == 0
           save_theta!(tr, name)
           n_snaps += 1
        end 
        
        nb += 1
    end

    #computing correlation matrix
    
    o = CovMatrix(n_snaps)
   
    rows = CSV.Rows("/mnt/Q_weights/"*name*".csv"; header = false)
    itr = (Float64.(collect(r)) for r in rows)
    fit!(o, itr)

    path = "/mnt/gram_stats/"
    if !isdir(path)
        mkpath(path)
    end

    file_path = joinpath(path, name * ".bson")
    BSON.@save file_path gram_matrix = Gr
    println("Gram stats saved to $file_path")

    return nothing
end


tr = load_trainer(name)
bf = load_buffer(name)

tr.buffer = bf

# Taking bigger batch size in order to have a better representation of the loss, it was 10_000
tr.buffer.batch_size = 256

compute_gram(tr, name)

