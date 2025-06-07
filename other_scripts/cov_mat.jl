#cov_mat3.jl, too slow, I will go with a diagonal hessian.
########################################################################
#I have a huge layer. I will split it in smaller parts 
using BSON, Flux, Zygote, Plots

include("imports.jl")

# ─── 1. Load trainer, buffer, network ─────────────────────────────────────
name   = "very_long_training1.bson"
folder = "./trainers/"
tr     = load_trainer(folder * name)
fill_buffer!(tr)

net    = tr.model.q_net
buffer = tr.buffer

# ─── 2. Compute one static batch (mutations OK here) ────────────────────
function get_batch_data(buffer)
    #buffer.batch_size = buffer.capacity
    buffer.batch_size = 100                     #trying to avoid to go out of memory
    batch = sample(buffer)
    return stack_exp(batch)  # returns (states, actions, rewards, next_states, dones, …)
end

states, actions, rewards, next_states, dones, _, _ = get_batch_data(buffer)
q_next     = tr.model.t_net(next_states)
max_next_q = dropdims(maximum(q_next, dims=1), dims=1)
q_target   = @. rewards + 0.99f0 * max_next_q * (1 - dones)

function get_idxs(model::Chain; block_size::Int = 1000)
    starts, ends = Int[], Int[]
    s = 0  # Current position in the flattened parameter vector

    for layer in model.layers
        ps = Flux.trainables(layer)
        isempty(ps) && continue

        total_len = sum(length(p) for p in ps)

        i = 0
        while i < total_len
            blk_len = min(block_size, total_len - i)
            push!(starts, s + 1)
            s += blk_len
            push!(ends, s)
            i += blk_len
        end
    end

    return starts, ends
end


start_idxs, end_idxs = get_idxs(net)

# ─── 4. Build a pure loss(w_flat) that uses our precomputed batch ───────
#    and closes over re, states, actions, q_target
w0, re = Flux.destructure(net)
function make_loss(states, actions, q_target, re)
    return function loss_fn(w_flat)
        model = re(w_flat)
        q_pred = model(states)                             # (n_actions, batch)
        # select the predicted Q for each taken action
        sel = [q_pred[a,i] for (i,a) in enumerate(actions)]
        return Flux.huber_loss(sel, q_target)
    end
end

loss_fn = make_loss(states, actions, q_target, re)

# pre-define this at top of file
# purely functional block replace—no mutation
function replace_block(w0::Vector{Float32}, idx::UnitRange{Int}, ws::Vector)
    # lift all Float32 into Dual{…}
    w_dual = w0 .+ zero(ws[1])  

    # split into before, slice, after
    i1 = idx.start
    i2 = idx.stop
    pre  = w_dual[1 : i1-1]
    post = w_dual[i2+1 : end]

    # concatenate pre, ws, post into a new Vector{Dual}
    return vcat(pre, ws, post)
end

#saving blocks.
function save_H_blocks(model::Chain, start_idxs, end_idxs, loss_fn)
    
    path = "./hessian_blocks/"
    if !isdir(path) mkpath(path) end
    
    w0, re = Flux.destructure(model)
    L = length(start_idxs)
    
    file = ".bson"
    
    for i in 1:L
        idx_i = start_idxs[i]:end_idxs[i]
        w_i   = w0[idx_i]

        H_i = Zygote.hessian(ws -> begin
            w_mod = replace_block(w0, idx_i, ws)
            loss_fn(w_mod)
        end, w_i)

        #saving 
        file = "hessian_block_" * string(i) * ".bson"
        @save file Hi
        @info "Saved hessian block $i"
    end

    return nothing
end

save_H_blocks(net, start_idxs, end_idxs, loss_fn)
