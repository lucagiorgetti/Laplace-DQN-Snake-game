#the replay buffer stores touples of the form (state, action, reward, next_state, done_or_not::Bool), position tracks the index where the next experience must be stored.
using StatsBase

#type alias
const Experience = Tuple{Matrix{Int64}, CartesianIndex{2}, Float64, Matrix{Int64}, Bool}


mutable struct ReplayBuffer
        capacity::Int
        position::Int
        buffer::Vector{Experience}
        
        function ReplayBuffer(capacity = 10000)
                  buffer = Vector{Experience}(undef, 0)
                  new(capacity, 1, buffer)
        end
end

#definition length of a buffer
Base.length(rpb::ReplayBuffer) = length(rpb.buffer)

function store!(rpb::ReplayBuffer, exp::Experience)
          if length(rpb) < rpb.capacity
              push!(rpb.buffer, exp)
          else
             rpb.buffer[rpb.position] = exp
             rpb.position += 1 
          end
          if rpb.position > rpb.capacity
              rpb.position = 1
          end
end

#I think sampling without replacement is faster because I do not want to repeat the initial state-actions for too many times.
function sample(rpb::ReplayBuffer, batch_size::Int)::Vector{Experience}
          if length(rpb) < batch_size
              return sample(rpb.buffer, length(rpb), replace = false)
          else
              return sample(rpb.buffer, batch_size, replace = false)
          end 
end

#-----------------------------------HELPER FUNCTIONS----------------------------------------------------------------------------------
function isfull(rpb::ReplayBuffer)
          return rpb.position == rpb.capacity
end

function isready(rpb::ReplayBuffer, batch_size::Int)
          return length(rpb) >= batch_size
end

function fill_buffer(rpb::ReplayBuffer, game::SnakeGame)
         while !isfull(rpb)
               # epsilon-greedy policy
               action = epsilon_greedy(game, model, epsilon = 1)
               exp = get_step(game, action)
               store!(rpb, exp)    
         end 
end 
