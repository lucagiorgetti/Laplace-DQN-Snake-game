include("imports.jl")

tr = Trainer(n_batches = 10000)
train!(tr, "short_training")

