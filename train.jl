#train loop
include("env.jl")
include("buffer.jl")
include("model.jl")

#start by acquiring experience until the buffer is full
game = SnakeGame
rpb = ReplayBuffer()
model = DQNModel(game)

while !isfull(rpb)
       # epsilon-greedy policy
       action = epsilon_greedy(game, model, epsilon = 1)
       
end 
