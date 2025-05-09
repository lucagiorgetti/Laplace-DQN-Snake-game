#main.jl
################################################################################################
include("imports.jl")

#setting DEBUG LEVEL
ENV["JULIA_DEBUG"] = Main

#setting up the logger
name = "food_placement2"
logger = SimpleLogger(open(name * ".txt", "w+"))
global_logger(logger)

tr = load_trainer("./trainers/short_training1.bson")
tr.buffer.capacity = 1000
fill_buffer!(tr)

