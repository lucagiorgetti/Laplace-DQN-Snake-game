#main.jl
################################################################################################
include("imports.jl")

#setting DEBUG LEVEL
ENV["JULIA_DEBUG"] = Main

#setting up the logger
name = "short_training3"
logger = SimpleLogger(open(name * ".txt", "w+"))
global_logger(logger)

tr = Trainer(n_batches = 1000)
train!(tr; trainer_name = name)

