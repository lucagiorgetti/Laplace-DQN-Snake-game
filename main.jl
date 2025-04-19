include("imports.jl")

game = SnakeGame()
rpb = ReplayBuffer()
model = DQNModel(game)

train!(model, rpb, 10, 5, 0.8f0)
