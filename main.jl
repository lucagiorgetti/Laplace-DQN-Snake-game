include("imports.jl")

tr = Trainer()
train!(tr, "short_training")
plot_loss(tr; mv_avg = true, save_name = "short_training")
