#main_la.jl
################################################################################################
include("imports.jl")

#setting DEBUG LEVEL
#ENV["JULIA_DEBUG"] = Main

#setting up the logger
name = "very_long_la_training5"
trainer_path = "./trainers/very_long_training1.bson"

io = open(name * ".txt", "w+")
logger = TerminalLogger(io;show_limited = false)
global_logger(logger)

#resume_training!(n_batches=100000, trainer_path=trainer_path, la_trainer_name=name)
resume_training_mod!(n_batches=100000, trainer_path=trainer_path, la_trainer_name=name)
flush(io)
close(io)
