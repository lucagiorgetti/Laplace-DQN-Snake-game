using Flux
using Flux.Optimise: RMSProp
include("env.jl")
include("buffer.jl")

mutable struct DQNModel
    q_net::Chain
    t_net::Chain
    opt::RMSProp

    function DQNModel(game::SnakeGame)
        board_size = game.state_size

        q_net = Chain(
            Conv((3, 3), 1 => 16, relu; pad=(1,1)), 
            Conv((3, 3), 16 => 32, relu; pad=(1,1)),
            Conv((6, 6), 32 => 64, relu),
            flatten,
            Dense(((board_size - 4) * (board_size - 4) * 64), 64, relu),
            Dense(64, 4)   #output n_actions == 4
        )

        t_net = deepcopy(q_net)
        opt = RMSProp(0.0005)   

        new(q_net, t_net, opt)
    end
end

#TODO:implement methods
function epsilon_greedy(game::SnakeGame, model::DQNModel, epsilon::Float64)::CartesianIndex{2}=0.01)

          actions = [CartesianIndex(1,0), CartesianIndex(-1,0), CartesianIndex(0,1), CartesianIndex(0,-1)]
          
          state = game.state
          
          if only(rand(1)) < epsilon
              act = rand(actions)
          else
             exp_rewards = model.q_network(state)
             max_idx = argmax(exp_rewards)
             act = actions[max_idx]
          end
          return act
end
