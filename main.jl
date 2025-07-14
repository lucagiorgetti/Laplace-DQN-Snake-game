#main.jl
################################################################################################
include("imports.jl")

#setting DEBUG LEVEL
ENV["JULIA_DEBUG"] = Main

#setting up the logger
name = "very_long_double_training2"
io = open(name * ".txt", "w+")
logger = TerminalLogger(io;show_limited = false)
global_logger(logger)

tr = Trainer(n_batches = 800000)
train!(tr; trainer_name = name)

flush(io)
close(io)
