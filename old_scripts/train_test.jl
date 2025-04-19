include("env.jl")      # defines SnakeGame
include("model.jl")    # defines DQNModel (needs SnakeGame)
include("buffer.jl")   # defines ReplayBuffer (needs DQNModel for fill_buffer!)
include("train.jl")    # defines train! (needs everything)

game = SnakeGame()
rpb = ReplayBuffer()
model = DQNModel(game)

train!(model, rpb, 10, 5, 0.8f0)
